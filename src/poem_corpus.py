"""Poem corpus: attach reference metadata + temporal plausibility + QC on top of an ExcerptUniverse.

This module owns **everything that depends on reference (CH + internet) catalogue
knowledge about individual poems**:

1. Cross-check of the ``poem_meta`` rollup (file's ``num_ppa_works`` vs distinct
   ``ppa_work_id`` recomputed from the excerpt frame).
2. Chadwyck canonicalisation: MD5-hash ``data/chadwyckhealey/<id>.txt`` and
   remap non-canonical poem_ids in excerpts to the canonical id in each group.
3. Missing-file report for Chadwyck normalized poem texts.
4. Reference catalogue build (CH via ``poetry_metadata.csv`` + internet via
   ``internet_poems_metadata_enriched.csv``) and a left-join onto the excerpt
   universe — this is what attaches ``ref_md_birth_year_wd``,
   ``ref_md_death_year_wd``, ``ref_md_ch_birth_lo``, ``ref_md_period`` …
5. Reference join coverage (which excerpt rows have a catalogue match).
6. Temporal plausibility **diagnostics** (per-corpus slice reports: how many
   rows would be dropped by the WD-birth / CH-birth / edition-floor rules).
   Diagnostics never drop rows.
7. Temporal plausibility **filter** (optional; applied *after* the diagnostics).
   This is the only stage that actually reduces the excerpt frame on
   poem-historical grounds.

Everything poem-historical lives here. PPA-side universe decisions (collection
filter, cluster collapse, poem_id blocklist) live in :mod:`excerpt_universe`
and are assumed already applied.

The builder is silent by default; use :func:`print_poem_corpus_summary` on the
returned bundle to print a human-readable rollup, or the individual ``print_*``
helpers for targeted deep-dives.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from excerpt_universe import ExcerptUniverse, n_poem_host_appearances
from ref_metadata import (
    build_reference_poem_metadata_df,
    join_reference_metadata_onto_excerpts,
    strip_unicode_bom_expr,
)

_CHADWYCK_TEXT_DIR = "chadwyckhealey"

# ─── small helpers ──────────────────────────────────────────────────────────


def poem_id_join_key_expr(col: pl.Expr) -> pl.Expr:
    """Normalise ``poem_id`` for joins (trim + strip BOM)."""
    return strip_unicode_bom_expr(col)


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _n_excerpt_unique(df: pl.DataFrame, mask: pl.Expr) -> int:
    return int(df.filter(mask).select(pl.col("excerpt_id").n_unique()).row(0)[0])


# ─── 1. poem_meta rollup QC ─────────────────────────────────────────────────


@dataclass(frozen=True)
class PoemMetaRollupQc:
    """Cross-check of ``num_ppa_works`` (poem_meta) vs recomputed distinct ``ppa_work_id``."""

    check: pl.DataFrame
    mismatches: pl.DataFrame
    n_pairs: int
    max_abs_delta: int | None


def qc_poem_meta_num_ppa_works(
    excerpts_df: pl.DataFrame, poem_meta_df: pl.DataFrame
) -> PoemMetaRollupQc:
    nonempty = poem_id_join_key_expr(pl.col("poem_id")) != ""
    recomputed = (
        excerpts_df.filter(nonempty)
        .with_columns(poem_id_join_key_expr(pl.col("poem_id")).alias("_poem_id_key"))
        .group_by("_poem_id_key", "ref_corpus")
        .agg(pl.col("ppa_work_id").n_unique().alias("n_works_recomputed"))
        .rename({"_poem_id_key": "poem_id"})
    )
    meta_sub = poem_meta_df.select(
        poem_id_join_key_expr(pl.col("poem_id")).alias("poem_id"),
        pl.col("ref_corpus").cast(pl.Utf8, strict=False).alias("ref_corpus"),
        pl.col("num_ppa_works").cast(pl.Int64, strict=False).alias("num_ppa_works_file"),
    )
    check = recomputed.join(meta_sub, on=["poem_id", "ref_corpus"], how="inner").with_columns(
        (pl.col("n_works_recomputed") - pl.col("num_ppa_works_file")).alias("delta_works")
    )
    mism = check.filter(pl.col("delta_works") != 0).sort(pl.col("delta_works").abs(), descending=True)
    max_abs = None
    if check.height:
        max_abs = int(check.select(pl.col("delta_works").abs().max()).to_series()[0])
    return PoemMetaRollupQc(check=check, mismatches=mism, n_pairs=check.height, max_abs_delta=max_abs)


# ─── 2. Chadwyck canonical id remap ─────────────────────────────────────────


@dataclass(frozen=True)
class ChCanonicalizeResult:
    excerpts_df: pl.DataFrame
    canonical_map: dict[str, str]
    n_ch_ids_checked: int
    n_files_missing: int
    n_dup_groups: int
    n_noncanonical_ids: int
    n_excerpt_rows_remapped: int
    dup_sample_rows: pl.DataFrame


def dedupe_chadwyck_poem_ids_by_normalized_text(
    excerpts_df: pl.DataFrame,
    poem_meta_df: pl.DataFrame,
    repo_root: Path,
) -> ChCanonicalizeResult:
    """Remap CH ``poem_id`` in excerpts to a canonical id when normalized text matches (MD5)."""
    norm_dir = repo_root / "data" / _CHADWYCK_TEXT_DIR
    ch_ids = (
        poem_meta_df.filter(pl.col("ref_corpus") == "chadwyck-healey")["poem_id"].to_list()
    )
    hash_to_ids: dict[str, list[str]] = {}
    n_missing = 0
    for pid in ch_ids:
        fp = norm_dir / f"{pid}.txt"
        if not fp.exists():
            n_missing += 1
            continue
        h = hashlib.md5(fp.read_bytes()).hexdigest()
        hash_to_ids.setdefault(h, []).append(pid)

    dup_groups = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}
    n_dup_groups = len(dup_groups)
    n_noncanonical = sum(len(ids) - 1 for ids in dup_groups.values())

    exc_counts: dict[str, int] = dict(
        zip(
            poem_meta_df["poem_id"].to_list(),
            poem_meta_df["num_excerpts"].cast(pl.Int64, strict=False).fill_null(0).to_list(),
        )
    )
    canonical_map: dict[str, str] = {}
    dup_rows: list[dict[str, Any]] = []
    for h, ids in list(dup_groups.items())[:50]:
        if len(dup_rows) >= 150:
            break
        canon = max(ids, key=lambda p: exc_counts.get(p, 0))
        sorted_ids = sorted(ids)
        for p in ids:
            if p != canon:
                canonical_map[p] = canon
            others = [x for x in sorted_ids if x != p]
            dup_rows.append(
                {
                    "md5_prefix": h[:12],
                    "poem_id": p,
                    "canonical_id": canon,
                    "is_canonical": p == canon,
                    "num_excerpts": exc_counts.get(p, 0),
                    "other_ids_in_group": others,
                    "n_ids_same_text": len(ids),
                }
            )
    dup_sample = pl.DataFrame(dup_rows) if dup_rows else pl.DataFrame(
        schema={
            "md5_prefix": pl.Utf8,
            "poem_id": pl.Utf8,
            "canonical_id": pl.Utf8,
            "is_canonical": pl.Boolean,
            "num_excerpts": pl.Int64,
            "other_ids_in_group": pl.List(pl.Utf8),
            "n_ids_same_text": pl.Int32,
        }
    )

    out = excerpts_df
    n_remapped = 0
    if canonical_map:
        remap_df = pl.DataFrame(
            {"poem_id_old": list(canonical_map.keys()), "poem_id_new": list(canonical_map.values())}
        )
        n_remapped = int(
            excerpts_df["poem_id"]
            .cast(pl.Utf8, strict=False)
            .is_in(list(canonical_map.keys()))
            .sum()
        )
        out = (
            excerpts_df.join(remap_df, left_on="poem_id", right_on="poem_id_old", how="left")
            .with_columns(
                pl.when(pl.col("poem_id_new").is_not_null())
                .then(pl.col("poem_id_new"))
                .otherwise(pl.col("poem_id"))
                .alias("poem_id")
            )
            .drop("poem_id_new")
        )

    return ChCanonicalizeResult(
        excerpts_df=out,
        canonical_map=canonical_map,
        n_ch_ids_checked=len(ch_ids),
        n_files_missing=n_missing,
        n_dup_groups=n_dup_groups,
        n_noncanonical_ids=n_noncanonical,
        n_excerpt_rows_remapped=n_remapped,
        dup_sample_rows=dup_sample,
    )


def chadwyck_missing_normalized_txt_report(
    poem_meta_df: pl.DataFrame, repo_root: Path, *, n_head: int = 20
) -> tuple[int, pl.DataFrame]:
    """Chadwyck ``poem_id`` values in ``poem_meta`` with no ``data/chadwyckhealey/<id>.txt`` file.

    Returns ``(n_missing, sample_df)`` where ``sample_df`` has up to ``n_head`` rows.
    """
    norm_dir = repo_root / "data" / _CHADWYCK_TEXT_DIR
    ch_ids = (
        poem_meta_df.filter(pl.col("ref_corpus") == "chadwyck-healey")["poem_id"]
        .cast(pl.Utf8, strict=False)
        .drop_nulls()
        .unique()
        .to_list()
    )
    missing = [pid for pid in ch_ids if str(pid).strip() and not (norm_dir / f"{pid}.txt").exists()]
    miss_df = pl.DataFrame({"poem_id": missing})
    if miss_df.height == 0:
        return 0, pl.DataFrame(
            schema={"poem_id": pl.Utf8, "author": pl.Utf8, "title": pl.Utf8},
        )
    meta_small = poem_meta_df.filter(pl.col("ref_corpus") == "chadwyck-healey").select(
        pl.col("poem_id").cast(pl.Utf8, strict=False).alias("poem_id"),
        pl.col("author").cast(pl.Utf8, strict=False).alias("author"),
        pl.col("title").cast(pl.Utf8, strict=False).alias("title"),
    )
    enriched = (
        miss_df.join(meta_small, on="poem_id", how="left")
        .group_by("poem_id", maintain_order=True)
        .agg(pl.col("author").first(), pl.col("title").first())
        .head(n_head)
    )
    return len(missing), enriched


# ─── 3. reference catalogue join coverage ──────────────────────────────────


@dataclass(frozen=True)
class ReferenceJoinCoverage:
    n_rows: int
    n_with_provenance: int
    pct_with_provenance: float
    by_corpus: pl.DataFrame
    orphan_sample: pl.DataFrame


def reference_join_coverage(excerpts_df: pl.DataFrame, k_orphan_sample: int = 25) -> ReferenceJoinCoverage:
    if "ref_md_provenance" not in excerpts_df.columns:
        empty = pl.DataFrame()
        return ReferenceJoinCoverage(
            n_rows=excerpts_df.height,
            n_with_provenance=0,
            pct_with_provenance=0.0,
            by_corpus=pl.DataFrame(
                schema={
                    "ref_corpus": pl.Utf8,
                    "n_rows": pl.UInt32,
                    "n_with_meta": pl.UInt32,
                    "pct_with_meta": pl.Float64,
                },
            ),
            orphan_sample=empty,
        )

    nonempty = (
        poem_id_join_key_expr(pl.col("poem_id")) != ""
    ) & pl.col("ref_corpus").is_not_null()
    base = excerpts_df.filter(nonempty)
    n_base = base.height
    with_meta = base.filter(pl.col("ref_md_provenance").is_not_null())
    n_wm = with_meta.height
    pct = 100.0 * n_wm / n_base if n_base else 0.0

    by_corpus = (
        base.group_by("ref_corpus")
        .agg(
            n_rows=pl.len(),
            n_with_meta=pl.col("ref_md_provenance").is_not_null().sum(),
        )
        .with_columns((100.0 * pl.col("n_with_meta") / pl.col("n_rows")).alias("pct_with_meta"))
        .sort("ref_corpus")
    )

    orphans = base.filter(pl.col("ref_md_provenance").is_null()).select(
        ["excerpt_id", "poem_id", "ref_corpus", "ppa_work_id", "ppa_pub_year", "poem_title"]
    )
    return ReferenceJoinCoverage(
        n_rows=n_base,
        n_with_provenance=n_wm,
        pct_with_provenance=pct,
        by_corpus=by_corpus,
        orphan_sample=orphans.head(k_orphan_sample),
    )


# ─── 4. temporal screening (per-corpus summary) ─────────────────────────────


@dataclass(frozen=True)
class TemporalCorpusSummary:
    """Per-``ref_corpus`` breakdown of what the temporal filter would do.

    All counts are on the *reference-joined* excerpt frame *before* the
    optional filter is applied. "Would drop" counts simulate the filter at the
    given ``poet_age_at_risk`` / ``poet_death_lookback`` parameters, regardless
    of whether the filter is actually enabled.
    """

    ref_corpus: str
    n_rows: int
    n_with_wd_birth: int
    n_with_ch_birth_only: int  # WD birth null, CH lower-bound present
    n_with_death_only: int  # both births null, death present
    n_no_bio: int  # rule B
    n_undated_host: int  # rule A
    # Strict (physically impossible) — host year strictly before WD birth year.
    n_impossible_wd_birth: int
    # Filter rules (what `_temporal_plausibility_mask` would drop in this corpus).
    n_would_drop_wd_birth: int
    n_would_drop_ch_birth: int
    n_would_drop_death: int
    n_would_drop_total: int
    # Diagnostic-only edition-floor signal (NOT part of the filter).
    n_edition_floor_only: int


def _summarise_temporal_by_corpus(
    df: pl.DataFrame, poet_age_at_risk: int, poet_death_lookback: int
) -> dict[str, TemporalCorpusSummary]:
    """Compute per-`ref_corpus` temporal summaries over the ref-joined excerpt frame."""
    if df.height == 0:
        return {}

    yhost = pl.col("ppa_pub_year").cast(pl.Int32, strict=False)
    b_wd = pl.col("ref_md_birth_year_wd").cast(pl.Int32, strict=False)
    b_ch = pl.col("ref_md_ch_birth_lo").cast(pl.Int32, strict=False)
    d_wd = pl.col("ref_md_death_year_wd").cast(pl.Int32, strict=False)
    ed = pl.col("ref_md_edition_floor_year").cast(pl.Int32, strict=False)

    has_wd = b_wd.is_not_null()
    has_ch_only = b_wd.is_null() & b_ch.is_not_null()
    has_death_only = b_wd.is_null() & b_ch.is_null() & d_wd.is_not_null()
    no_bio = b_wd.is_null() & b_ch.is_null() & d_wd.is_null()
    undated = yhost.is_null()

    impossible_wd = has_wd & yhost.is_not_null() & (yhost < b_wd)

    drop_wd = has_wd & yhost.is_not_null() & (yhost < b_wd + poet_age_at_risk)
    drop_ch = has_ch_only & yhost.is_not_null() & (yhost < b_ch + poet_age_at_risk)
    drop_death = has_death_only & yhost.is_not_null() & (yhost < d_wd - poet_death_lookback)
    drop_any = drop_wd | drop_ch | drop_death

    # Edition-floor flag that is *not* already captured by any filter rule.
    ed_flag = ed.is_not_null() & yhost.is_not_null() & (yhost < ed)
    ed_only = ed_flag & ~drop_any

    tagged = df.with_columns(
        _has_wd=has_wd,
        _has_ch_only=has_ch_only,
        _has_death_only=has_death_only,
        _no_bio=no_bio,
        _undated=undated,
        _impossible_wd=impossible_wd,
        _drop_wd=drop_wd,
        _drop_ch=drop_ch,
        _drop_death=drop_death,
        _drop_any=drop_any,
        _ed_only=ed_only,
    )

    out: dict[str, TemporalCorpusSummary] = {}
    for rc in tagged["ref_corpus"].unique().drop_nulls().to_list():
        sub = tagged.filter(pl.col("ref_corpus") == rc)
        out[rc] = TemporalCorpusSummary(
            ref_corpus=rc,
            n_rows=sub.height,
            n_with_wd_birth=int(sub["_has_wd"].sum()),
            n_with_ch_birth_only=int(sub["_has_ch_only"].sum()),
            n_with_death_only=int(sub["_has_death_only"].sum()),
            n_no_bio=int(sub["_no_bio"].sum()),
            n_undated_host=int(sub["_undated"].sum()),
            n_impossible_wd_birth=int(sub["_impossible_wd"].sum()),
            n_would_drop_wd_birth=int(sub["_drop_wd"].sum()),
            n_would_drop_ch_birth=int(sub["_drop_ch"].sum()),
            n_would_drop_death=int(sub["_drop_death"].sum()),
            n_would_drop_total=int(sub["_drop_any"].sum()),
            n_edition_floor_only=int(sub["_ed_only"].sum()),
        )
    return out


# ─── 5. temporal plausibility FILTER ────────────────────────────────────────


@dataclass(frozen=True)
class TemporalFilterReport:
    """What the temporal plausibility filter would drop (or did drop)."""

    applied: bool
    params: dict[str, Any]
    n_before: int
    n_after: int
    n_dropped_rule_c_wd_birth: int
    n_dropped_rule_d_ch_birth: int
    n_dropped_rule_e_death: int
    n_dropped_total: int
    n_appearances_before: int
    n_appearances_after: int
    n_appearances_dropped: int
    n_distinct_host_works_before: int
    n_distinct_host_works_after: int
    n_with_wd_birth: int
    n_with_ch_birth_only: int
    n_with_death_only: int
    n_with_no_bio: int
    top_dropped_poems: pl.DataFrame


def _temporal_plausibility_mask(
    poet_age_at_risk: int, poet_death_lookback: int
) -> pl.Expr:
    """Return a boolean expression: True = keep, False = temporally implausible.

    Priority cascade (first rule that applies wins):
        A. undated host → keep
        B. no bio signal at all → keep
        C. WD birth year available → host ≥ birth + poet_age_at_risk
        D. WD birth null, CH birth lo available → host ≥ ch_birth_lo + poet_age_at_risk
        E. both births null, death year available → host ≥ death − poet_death_lookback
    """
    yhost = pl.col("ppa_pub_year").cast(pl.Int32, strict=False)
    b_wd = pl.col("ref_md_birth_year_wd").cast(pl.Int32, strict=False)
    b_ch = pl.col("ref_md_ch_birth_lo").cast(pl.Int32, strict=False)
    d_wd = pl.col("ref_md_death_year_wd").cast(pl.Int32, strict=False)

    rule_a = yhost.is_null()
    rule_b = b_wd.is_null() & b_ch.is_null() & d_wd.is_null()
    rule_c = b_wd.is_not_null() & (yhost >= b_wd + poet_age_at_risk)
    rule_d = b_wd.is_null() & b_ch.is_not_null() & (yhost >= b_ch + poet_age_at_risk)
    rule_e = b_wd.is_null() & b_ch.is_null() & d_wd.is_not_null() & (
        yhost >= d_wd - poet_death_lookback
    )
    return rule_a | rule_b | rule_c | rule_d | rule_e


def _n_distinct_ppa_work_id(df: pl.DataFrame) -> int:
    """Count distinct ``ppa_work_id`` in excerpt rows (host-work breadth in the alignment frame)."""
    if df.height == 0:
        return 0
    return int(df.select(pl.col("ppa_work_id").n_unique()).row(0)[0])


def _temporal_filter_report(
    df_before: pl.DataFrame,
    df_after: pl.DataFrame,
    applied: bool,
    poet_age_at_risk: int,
    poet_death_lookback: int,
) -> TemporalFilterReport:
    yhost = pl.col("ppa_pub_year").cast(pl.Int32, strict=False)
    b_wd = pl.col("ref_md_birth_year_wd").cast(pl.Int32, strict=False)
    b_ch = pl.col("ref_md_ch_birth_lo").cast(pl.Int32, strict=False)
    d_wd = pl.col("ref_md_death_year_wd").cast(pl.Int32, strict=False)

    has_b_wd = b_wd.is_not_null()
    has_b_ch = b_wd.is_null() & b_ch.is_not_null()
    has_d_wd = b_wd.is_null() & b_ch.is_null() & d_wd.is_not_null()

    impl_c = has_b_wd & (yhost < b_wd + poet_age_at_risk)
    impl_d = has_b_ch & (yhost < b_ch + poet_age_at_risk)
    impl_e = has_d_wd & (yhost < d_wd - poet_death_lookback)
    impl_any = impl_c | impl_d | impl_e

    tagged = df_before.with_columns(
        _impl_c=impl_c,
        _impl_d=impl_d,
        _impl_e=impl_e,
        _impl_any=impl_any,
        _has_b_wd=has_b_wd,
        _has_b_ch=has_b_ch,
        _has_d_wd=has_d_wd,
        _no_bio=b_wd.is_null() & b_ch.is_null() & d_wd.is_null(),
    )
    _dropped = tagged.filter(pl.col("_impl_any"))
    # aggregate per poem, carrying title + author when present on excerpts_df
    _agg_cols = [pl.len().alias("n_dropped")]
    if "poem_title" in _dropped.columns:
        _agg_cols.append(pl.col("poem_title").drop_nulls().first().alias("poem_title"))
    if "poem_author" in _dropped.columns:
        _agg_cols.append(pl.col("poem_author").drop_nulls().first().alias("poem_author"))
    top_dropped = (
        _dropped.group_by("poem_id")
        .agg(_agg_cols)
        .sort("n_dropped", descending=True)
        .head(10)
    )
    app_before = n_poem_host_appearances(df_before)
    app_after = n_poem_host_appearances(df_after)
    hosts_before = _n_distinct_ppa_work_id(df_before)
    hosts_after = _n_distinct_ppa_work_id(df_after)
    return TemporalFilterReport(
        applied=applied,
        params={
            "poet_age_at_risk": poet_age_at_risk,
            "poet_death_lookback": poet_death_lookback,
        },
        n_before=df_before.height,
        n_after=df_after.height,
        n_dropped_rule_c_wd_birth=int(tagged["_impl_c"].sum()),
        n_dropped_rule_d_ch_birth=int(tagged["_impl_d"].sum()),
        n_dropped_rule_e_death=int(tagged["_impl_e"].sum()),
        n_dropped_total=df_before.height - df_after.height,
        n_appearances_before=app_before,
        n_appearances_after=app_after,
        n_appearances_dropped=app_before - app_after,
        n_distinct_host_works_before=hosts_before,
        n_distinct_host_works_after=hosts_after,
        n_with_wd_birth=int(tagged["_has_b_wd"].sum()),
        n_with_ch_birth_only=int(tagged["_has_b_ch"].sum()),
        n_with_death_only=int(tagged["_has_d_wd"].sum()),
        n_with_no_bio=int(tagged["_no_bio"].sum()),
        top_dropped_poems=top_dropped,
    )


def _compute_flagged_poem_ids(
    excerpts_df: pl.DataFrame,
    poet_age_at_risk: int,
    poet_death_lookback: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return three poem-level flag artifacts.

    For each poem, we report how many rows trigger each temporal flag:

    - **filter rules** — rows the temporal filter *would* drop at the given
      ``poet_age_at_risk`` / ``poet_death_lookback`` (WD birth, CH lower-bound
      birth, or death-only rules).
    - **edition-floor only** — rows flagged because ``ppa_pub_year`` precedes
      ``ref_md_edition_floor_year`` but NOT caught by any filter rule. The
      temporal filter ignores this signal because edition-floor year describes
      the catalog witness, not the original composition.
    - **any flag** — union of the two (superset; backwards-compatible).

    Returns ``(any_flag, filter_rules, edition_floor_only)``.
    """
    yhost = pl.col("ppa_pub_year").cast(pl.Int32, strict=False)
    b_wd = pl.col("ref_md_birth_year_wd").cast(pl.Int32, strict=False)
    b_ch = pl.col("ref_md_ch_birth_lo").cast(pl.Int32, strict=False)
    d_wd = pl.col("ref_md_death_year_wd").cast(pl.Int32, strict=False)
    ed = pl.col("ref_md_edition_floor_year").cast(pl.Int32, strict=False)

    # Filter-rule flags (what the filter would drop).
    drop_wd = b_wd.is_not_null() & yhost.is_not_null() & (yhost < b_wd + poet_age_at_risk)
    drop_ch = (
        b_wd.is_null()
        & b_ch.is_not_null()
        & yhost.is_not_null()
        & (yhost < b_ch + poet_age_at_risk)
    )
    drop_death = (
        b_wd.is_null()
        & b_ch.is_null()
        & d_wd.is_not_null()
        & yhost.is_not_null()
        & (yhost < d_wd - poet_death_lookback)
    )
    drop_any = drop_wd | drop_ch | drop_death

    # Edition-floor flag (diagnostic only).
    ed_flag = ed.is_not_null() & yhost.is_not_null() & (yhost < ed)
    ed_only = ed_flag & ~drop_any

    tagged = excerpts_df.with_columns(
        _drop_wd=drop_wd,
        _drop_ch=drop_ch,
        _drop_death=drop_death,
        _drop_any=drop_any,
        _ed_flag=ed_flag,
        _ed_only=ed_only,
    )

    # Poems caught by filter rules (what gets dropped).
    filter_rules = (
        tagged.filter(pl.col("_drop_any"))
        .group_by(["poem_id", "ref_corpus"])
        .agg(
            n_rows_would_drop=pl.len(),
            n_rows_wd_birth=pl.col("_drop_wd").cast(pl.UInt32).sum(),
            n_rows_ch_birth=pl.col("_drop_ch").cast(pl.UInt32).sum(),
            n_rows_death_only=pl.col("_drop_death").cast(pl.UInt32).sum(),
            poem_title=pl.col("poem_title").drop_nulls().first(),
            poem_author=pl.col("poem_author").drop_nulls().first(),
            ref_md_birth_year_wd=pl.col("ref_md_birth_year_wd").drop_nulls().first(),
            ref_md_catalog_title=pl.col("ref_md_catalog_title").drop_nulls().first(),
        )
        .sort("n_rows_would_drop", descending=True)
    )

    # Poems where ONLY the edition-floor signal fires (not filtered — diagnostic).
    edition_floor_only = (
        tagged.filter(pl.col("_ed_only"))
        .group_by(["poem_id", "ref_corpus"])
        .agg(
            n_rows_edition_floor_only=pl.len(),
            poem_title=pl.col("poem_title").drop_nulls().first(),
            poem_author=pl.col("poem_author").drop_nulls().first(),
            ref_md_edition_floor_year=pl.col("ref_md_edition_floor_year").drop_nulls().first(),
            ref_md_birth_year_wd=pl.col("ref_md_birth_year_wd").drop_nulls().first(),
            ref_md_catalog_title=pl.col("ref_md_catalog_title").drop_nulls().first(),
        )
        .sort("n_rows_edition_floor_only", descending=True)
    )

    # Union (superset) — kept for backward-compatible `flagged_poem_ids` attribute.
    any_flag_expr = pl.col("_drop_any") | pl.col("_ed_flag")
    any_flag = (
        tagged.filter(any_flag_expr)
        .group_by(["poem_id", "ref_corpus"])
        .agg(
            n_rows_any_flag=pl.len(),
            n_rows_would_drop=pl.col("_drop_any").cast(pl.UInt32).sum(),
            n_rows_edition_floor=pl.col("_ed_flag").cast(pl.UInt32).sum(),
            n_rows_wd_birth=pl.col("_drop_wd").cast(pl.UInt32).sum(),
            n_rows_ch_birth=pl.col("_drop_ch").cast(pl.UInt32).sum(),
            n_rows_death_only=pl.col("_drop_death").cast(pl.UInt32).sum(),
            poem_title=pl.col("poem_title").drop_nulls().first(),
            poem_author=pl.col("poem_author").drop_nulls().first(),
            ref_md_birth_year_wd=pl.col("ref_md_birth_year_wd").drop_nulls().first(),
            ref_md_catalog_title=pl.col("ref_md_catalog_title").drop_nulls().first(),
        )
        .sort("n_rows_any_flag", descending=True)
    )

    return any_flag, filter_rules, edition_floor_only


# ─── 6. public bundle + builder ─────────────────────────────────────────────


@dataclass(frozen=True)
class PoemCorpus:
    """Analysis-ready excerpt frame plus all poem-side QC artifacts."""

    # data (active = after any optional temporal filter)
    excerpts_df: pl.DataFrame
    excerpts_df_prefilter: pl.DataFrame  # canonicalised + ref-joined, no temporal filter
    reference_poem_metadata_df: pl.DataFrame
    poem_meta_df: pl.DataFrame
    poetry_meta_poem_df: pl.DataFrame  # CH-only slice of reference metadata

    # QC artifacts
    rollup_qc: PoemMetaRollupQc
    canonicalization: ChCanonicalizeResult
    missing_norm_txt: tuple[int, pl.DataFrame]
    reference_coverage: ReferenceJoinCoverage

    # Unified temporal screening (merged diagnostics + filter view)
    temporal_by_corpus: dict[str, TemporalCorpusSummary]
    temporal_filter: TemporalFilterReport

    # Poem-level flag artifacts
    flagged_by_filter_rules: pl.DataFrame       # poems the filter would / did drop rows for
    flagged_by_edition_floor_only: pl.DataFrame  # poems flagged only by edition-floor signal (NOT filtered)
    flagged_poem_ids: pl.DataFrame               # backwards-compatible union superset

    # configuration (echoed back for downstream use)
    apply_temporal_filter: bool
    poet_age_at_risk: int
    poet_death_lookback: int


def build_poem_corpus(
    universe: ExcerptUniverse,
    *,
    repo_root: Path,
    poem_meta_path: Path,
    apply_temporal_filter: bool = True,
    poet_age_at_risk: int = 18,
    poet_death_lookback: int = 60,
) -> PoemCorpus:
    """Attach reference metadata + QC to an ExcerptUniverse; optionally drop temporally implausible rows.

    Silent by default. Use :func:`print_poem_corpus_summary` on the returned
    bundle to emit a human-readable report.
    """
    excerpts_df = universe.excerpts_df

    # Load the poem_meta table (sidecar to the Passim excerpts — not to be
    # confused with the richer reference catalogue built below).
    poem_meta_df = pl.read_csv(poem_meta_path)
    _bom = chr(0xFEFF)
    poem_meta_df = poem_meta_df.rename(
        {c: c.lstrip(_bom) for c in poem_meta_df.columns if c.startswith(_bom)}
    )

    # 1. rollup QC
    rollup_qc = qc_poem_meta_num_ppa_works(excerpts_df, poem_meta_df)

    # 2. Chadwyck canonicalisation
    canon = dedupe_chadwyck_poem_ids_by_normalized_text(excerpts_df, poem_meta_df, repo_root)
    excerpts_df = canon.excerpts_df

    # 3. missing-file report
    missing = chadwyck_missing_normalized_txt_report(poem_meta_df, repo_root)

    # 4. reference catalogue build + join
    reference_poem_metadata_df, poetry_meta_poem_df = build_reference_poem_metadata_df(
        repo_root, canonical_map=canon.canonical_map
    )
    excerpts_df = join_reference_metadata_onto_excerpts(excerpts_df, reference_poem_metadata_df)

    # 5. reference join coverage
    coverage = reference_join_coverage(excerpts_df)

    # 6. unified per-ref_corpus temporal summary (what the filter *would* drop).
    temporal_by_corpus = _summarise_temporal_by_corpus(
        excerpts_df, poet_age_at_risk, poet_death_lookback
    )

    # 7. per-poem flag artifacts: split into filter-rule and edition-floor-only.
    any_flag_df, filter_rules_df, edition_floor_only_df = _compute_flagged_poem_ids(
        excerpts_df,
        poet_age_at_risk=poet_age_at_risk,
        poet_death_lookback=poet_death_lookback,
    )

    # 8. optional temporal plausibility filter.
    prefilter = excerpts_df
    if apply_temporal_filter:
        mask = _temporal_plausibility_mask(poet_age_at_risk, poet_death_lookback)
        excerpts_df = excerpts_df.filter(mask)
    t_report = _temporal_filter_report(
        prefilter, excerpts_df, applied=apply_temporal_filter,
        poet_age_at_risk=poet_age_at_risk, poet_death_lookback=poet_death_lookback,
    )

    return PoemCorpus(
        excerpts_df=excerpts_df,
        excerpts_df_prefilter=prefilter,
        reference_poem_metadata_df=reference_poem_metadata_df,
        poem_meta_df=poem_meta_df,
        poetry_meta_poem_df=poetry_meta_poem_df,
        rollup_qc=rollup_qc,
        canonicalization=canon,
        missing_norm_txt=missing,
        reference_coverage=coverage,
        temporal_by_corpus=temporal_by_corpus,
        temporal_filter=t_report,
        flagged_by_filter_rules=filter_rules_df,
        flagged_by_edition_floor_only=edition_floor_only_df,
        flagged_poem_ids=any_flag_df,
        apply_temporal_filter=apply_temporal_filter,
        poet_age_at_risk=poet_age_at_risk,
        poet_death_lookback=poet_death_lookback,
    )


# ─── 7. summary printers ────────────────────────────────────────────────────


def _truncate(s: object, n: int) -> str:
    t = "" if s is None else str(s)
    return t if len(t) <= n else t[: n - 1] + "…"


def print_rollup_qc(c: PoemCorpus) -> None:
    """Universe coverage vs upstream manifest (informational sanity check)."""
    r = c.rollup_qc
    print("[1] Universe coverage vs upstream manifest")
    print(
        "    For each poem, compare two counts of distinct host works:\n"
        "      file       = num_ppa_works in poem_meta.csv (original release, pre-filter)\n"
        "      recomputed = distinct ppa_work_id we actually see in the filtered universe\n"
        "      delta      = recomputed − file  (≈ 0 with no filters; large negative with filters)"
    )
    print(
        f"    {r.n_pairs:,} (poem_id, ref_corpus) pairs matched in both | "
        f"max |Δn_works| = {r.max_abs_delta}"
    )
    if r.mismatches.height:
        print(
            f"    {r.mismatches.height:,} pairs with non-zero delta "
            "(expected — this is how we know the host-side filters fired)."
        )


def print_canonicalization(c: PoemCorpus) -> None:
    ch = c.canonicalization
    print("[2] Chadwyck canonical-id dedupe (MD5 of data/chadwyckhealey/<id>.txt)")
    print(f"    ids considered       : {ch.n_ch_ids_checked:>8,}")
    print(f"    files missing on disk: {ch.n_files_missing:>8,}")
    print(f"    duplicate groups     : {ch.n_dup_groups:>8,}")
    print(f"    secondary remapped   : {ch.n_noncanonical_ids:>8,}")
    print(f"    excerpt rows remapped: {ch.n_excerpt_rows_remapped:>8,}")


def print_missing_norm_txt(c: PoemCorpus, n_head: int = 20) -> None:
    n_missing, sample = c.missing_norm_txt
    if n_missing == 0:
        print("[2b] Missing chadwyckhealey text files: none")
        return
    print(f"[2b] Chadwyck poem_ids with no data/chadwyckhealey/<id>.txt: {n_missing:,}")
    if n_head > 0 and sample.height:
        print(sample.head(n_head))


def print_reference_coverage(c: PoemCorpus) -> None:
    cov = c.reference_coverage
    print("[3] Reference catalogue join coverage (CH + internet_poems ref_md_*)")
    print(
        f"    {cov.n_with_provenance:,} / {cov.n_rows:,} rows have catalogue match "
        f"({cov.pct_with_provenance:.2f}%)"
    )
    for row in cov.by_corpus.iter_rows(named=True):
        print(
            f"      {str(row['ref_corpus']):<20s} "
            f"rows={row['n_rows']:>8,}  with_meta={row['n_with_meta']:>8,}  "
            f"({row['pct_with_meta']:.2f}%)"
        )


def print_temporal_screening(c: PoemCorpus) -> None:
    """Unified screening report: rules, per-corpus breakdown, and filter outcome."""
    t = c.temporal_filter
    age = t.params["poet_age_at_risk"]
    lookback = t.params["poet_death_lookback"]

    status = "applied" if t.applied else "SKIPPED (apply_temporal_filter=False)"
    print(f"[4] Temporal screening — filter {status}")
    print(
        f"    Rules (cascade; first applicable wins):\n"
        f"      A. undated host → keep\n"
        f"      B. no bio anchor (WD birth / CH birth / WD death all null) → keep\n"
        f"      C. WD birth present     → drop if host_year < wd_birth + {age}\n"
        f"      D. WD birth null, CH birth present → drop if host_year < ch_birth + {age}\n"
        f"      E. only WD death        → drop if host_year < wd_death − {lookback}\n"
        f"    params: poet_age_at_risk={age}, poet_death_lookback={lookback}"
    )

    if c.temporal_by_corpus:
        print("\n    Per-ref_corpus breakdown (on the ref-joined frame before filtering):")
        print(
            f"      {'ref_corpus':<18s}  {'rows':>9s}  {'WD':>6s} {'CHonly':>7s} {'death':>6s}  "
            f"{'impossible':>10s}  {'wouldDrop':>10s}  {'edOnly':>7s}"
        )
        print(
            f"      {'-' * 18}  {'-' * 9}  {'-' * 6} {'-' * 7} {'-' * 6}  {'-' * 10}  "
            f"{'-' * 10}  {'-' * 7}"
        )
        for rc in sorted(c.temporal_by_corpus):
            s = c.temporal_by_corpus[rc]
            drop_pct = _pct(s.n_would_drop_total, s.n_rows)
            print(
                f"      {s.ref_corpus:<18s}  {s.n_rows:>9,}  "
                f"{s.n_with_wd_birth:>6,} {s.n_with_ch_birth_only:>7,} {s.n_with_death_only:>6,}  "
                f"{s.n_impossible_wd_birth:>10,}  "
                f"{s.n_would_drop_total:>10,}  {s.n_edition_floor_only:>7,}"
            )
            print(
                f"      {'':<18s}  {'':>9s}  "
                f"                          → of those: drop-wd={s.n_would_drop_wd_birth:,} "
                f"drop-ch={s.n_would_drop_ch_birth:,} drop-death={s.n_would_drop_death:,}  "
                f"({drop_pct:.2f}% of rows)"
            )
        print(
            "\n      Legend:\n"
            "        WD / CHonly / death = rows where each anchor is the applicable rule\n"
            "        impossible          = host_year < wd_birth (strict; informational)\n"
            "        wouldDrop           = rows the filter would drop (WD + CH + death rules)\n"
            "        edOnly              = rows flagged only by ref_md_edition_floor_year\n"
            "                              (catalog witness signal; NOT applied to the filter)"
        )

    if not t.applied:
        would_drop = (
            t.n_dropped_rule_c_wd_birth
            + t.n_dropped_rule_d_ch_birth
            + t.n_dropped_rule_e_death
        )
        print(
            f"\n    Filter not applied. Under current params it would drop {would_drop:,} "
            f"rows from the full {t.n_before:,}."
        )
        print(
            f"    Distinct poem–host appearances (same frame): "
            f"{t.n_appearances_before:,} (unchanged; filter off)."
        )
        print(
            f"    Distinct PPA host works (ppa_work_id in excerpts): "
            f"{t.n_distinct_host_works_before:,} (unchanged; filter off)."
        )
        return

    print(
        f"\n    Filter outcome: {t.n_before:,} → {t.n_after:,} rows "
        f"(dropped {t.n_dropped_total:,}, {_pct(t.n_dropped_total, t.n_before):.2f}%)"
    )
    print(
        f"    appearances : {t.n_appearances_before:,} → {t.n_appearances_after:,} "
        f"(dropped {t.n_appearances_dropped:,}, {_pct(t.n_appearances_dropped, t.n_appearances_before):.2f}%)"
    )
    hosts_lost = t.n_distinct_host_works_before - t.n_distinct_host_works_after
    print(
        f"    host works  : {t.n_distinct_host_works_before:,} → "
        f"{t.n_distinct_host_works_after:,} "
        f"(lost {hosts_lost:,}, {_pct(hosts_lost, t.n_distinct_host_works_before):.2f}%)"
    )
    print(f"      by WD birth rule: {t.n_dropped_rule_c_wd_birth:>7,}")
    print(f"      by CH birth rule: {t.n_dropped_rule_d_ch_birth:>7,}")
    print(f"      by death rule   : {t.n_dropped_rule_e_death:>7,}")
    print(
        f"      kept (rule A/B) : {t.n_with_no_bio:,} no-bio rows "
        "(plus all rows with an anchor that passed its rule)"
    )

    if t.top_dropped_poems.height:
        print("\n    Top poems by rows dropped (author / title when known):")
        for row in t.top_dropped_poems.head(10).iter_rows(named=True):
            pid = _truncate(row.get("poem_id"), 32)
            title = _truncate(row.get("poem_title"), 38)
            author = _truncate(row.get("poem_author"), 26)
            n = row.get("n_dropped", 0)
            print(f"      {pid:<32s}  {n:>6,} rows  | {author:<26s} | {title}")


def print_poem_corpus_summary(c: PoemCorpus) -> None:
    """One-stop summary of the poem-side pipeline (QC → ref join → temporal screening)."""
    print("Poem corpus\n" + "─" * 72)
    print_rollup_qc(c)
    print()
    print_canonicalization(c)
    print()
    print_missing_norm_txt(c, n_head=0)
    print()
    print_reference_coverage(c)
    print()
    print_temporal_screening(c)
    print()
    print("[5] Final active analysis frame")
    print(f"    corpus.excerpts_df shape : {c.excerpts_df.shape}")
    print(
        f"    distinct poem–host pairs : {n_poem_host_appearances(c.excerpts_df):>10,}  "
        "(unique poem_id ∧ ppa_work_id)"
    )
    print(
        f"    flagged_by_filter_rules       : {c.flagged_by_filter_rules.height:,} poems  "
        "(poems with ≥1 row the filter would drop)"
    )
    print(
        f"    flagged_by_edition_floor_only : {c.flagged_by_edition_floor_only.height:,} poems  "
        "(diagnostic only — NOT removed by the filter)"
    )
    print(
        f"    flagged_poem_ids              : {c.flagged_poem_ids.height:,} poems  "
        "(union of both; backwards-compatible superset)"
    )

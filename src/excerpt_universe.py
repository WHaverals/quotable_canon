"""Passim excerpt universe: raw load + host-side filtering + exposure tables.

This module defines the *host-opportunity universe* used by all downstream
modeling:

1. Load raw Passim excerpts and attach host metadata + lightweight poem sidecar.
2. Apply host-side filters (collection tags, cluster collapse, poem-id blocklist).
3. Build exposure tables from the filtered host metadata.

Nothing poem-historical happens here (reference-catalogue joins, temporal
plausibility). That is handled in :mod:`poem_corpus`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import polars as pl

ExpoDenom = Literal["total_pages", "n_works"]
CollectionFilterMode = Literal["any", "all"]


# ──────────────────────────────────────────────────────────────────────────────
# Raw load
# ──────────────────────────────────────────────────────────────────────────────


def _load_raw_excerpts(
    excerpts_path: Path,
    ppa_work_metadata_path: Path,
    poem_meta_path: Path,
) -> pl.DataFrame:
    """Load raw Passim excerpts, attach host metadata + poem sidecar fields."""
    from corppa.poetry_detection.polars_utils import (
        add_ppa_works_meta,
        extract_page_meta,
        load_excerpts_df,
    )

    _ex = load_excerpts_df(excerpts_path, ppa_works_meta=None, ref_poems_meta=None)
    _ex = extract_page_meta(_ex)
    _with_ppa = add_ppa_works_meta(_ex, ppa_work_metadata_path)

    _pm = pl.read_csv(poem_meta_path, infer_schema_length=10_000)
    _bom = chr(0xFEFF)
    _pm = _pm.rename({c: c.lstrip(_bom) for c in _pm.columns if c.startswith(_bom)})
    _poem_side = _pm.select(
        pl.col("poem_id"),
        pl.col("ref_corpus"),
        pl.col("author").alias("poem_author"),
        pl.col("title").alias("poem_title"),
        pl.col("num_lines").cast(pl.Int32, strict=False).alias("poem_num_lines"),
        pl.col("num_words").cast(pl.Int32, strict=False).alias("poem_num_words"),
        pl.col("char_len").cast(pl.Int32, strict=False).alias("poem_char_len"),
    )
    excerpts_df = _with_ppa.join(_poem_side, on=["poem_id", "ref_corpus"], how="left")

    if "ppa_pub_year" not in excerpts_df.columns and "ppa_work_year" in excerpts_df.columns:
        excerpts_df = excerpts_df.rename({"ppa_work_year": "ppa_pub_year"})

    return excerpts_df.with_columns(
        ppa_pub_decade=pl.col("ppa_pub_year").cast(pl.Int32, strict=False).floordiv(10).mul(10)
    ).cast(
        {"poem_num_lines": pl.Int32, "poem_num_words": pl.Int32, "poem_char_len": pl.Int32},
        strict=False,
    )


def _load_ppa_work_metadata(ppa_work_metadata_path: Path) -> pl.DataFrame:
    return pl.read_csv(ppa_work_metadata_path)


# ──────────────────────────────────────────────────────────────────────────────
# Collection filtering
# ──────────────────────────────────────────────────────────────────────────────


def n_poem_host_appearances(df: pl.DataFrame) -> int:
    """Count distinct (poem_id, ppa_work_id) pairs — one row per poem appearance in a host work.

    Alignment rows can repeat the same appearance; this is the deduped unit for breadth-style stats.
    """
    if df.height == 0:
        return 0
    return int(df.select(["poem_id", "ppa_work_id"]).unique().height)


def _normalize_collection_filter(collection_filter: str | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize collection_filter input to a unique token tuple."""
    if collection_filter is None:
        return ()
    if isinstance(collection_filter, str):
        raw = [collection_filter]
    else:
        raw = list(collection_filter)

    out: list[str] = []
    for tok in raw:
        t = str(tok).strip()
        if t and t not in out:
            out.append(t)
    return tuple(out)


def _collection_filter_mask(
    col_name: str,
    tags: tuple[str, ...],
    mode: CollectionFilterMode,
) -> pl.Expr:
    """Filter expression for semicolon-delimited collection tags."""
    if not tags:
        return pl.lit(True)
    tokens = (
        pl.col(col_name)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.split(";")
        .list.eval(pl.element().str.strip_chars())
    )
    checks = [tokens.list.contains(tag) for tag in tags]
    expr = checks[0]
    for chk in checks[1:]:
        expr = (expr & chk) if mode == "all" else (expr | chk)
    return expr.fill_null(False)


# ──────────────────────────────────────────────────────────────────────────────
# Cluster collapse
# ──────────────────────────────────────────────────────────────────────────────


def _cluster_min_year(df: pl.DataFrame, cluster_col: str, year_col: str) -> pl.DataFrame:
    _cid = pl.col(cluster_col).cast(pl.Utf8, strict=False).fill_null("").str.strip_chars()
    _year = pl.col(year_col).cast(pl.Int32, strict=False)
    _has_cid = _cid != ""
    return (
        df.filter(_has_cid & _year.is_not_null())
        .group_by(cluster_col)
        .agg(_year.min().alias("_cluster_min_pub_year"))
    )


def _apply_cluster_collapse_excerpts(df: pl.DataFrame) -> pl.DataFrame:
    _cid = pl.col("ppa_cluster_id").cast(pl.Utf8, strict=False).fill_null("").str.strip_chars()
    _year = pl.col("ppa_pub_year").cast(pl.Int32, strict=False)
    _has_cid = _cid != ""
    mins = _cluster_min_year(df, "ppa_cluster_id", "ppa_pub_year")
    merged = df.join(mins, on="ppa_cluster_id", how="left")
    return merged.filter(
        (~_has_cid) | pl.col("_cluster_min_pub_year").is_null() | (_year == pl.col("_cluster_min_pub_year"))
    ).drop("_cluster_min_pub_year")


def _apply_cluster_collapse_work_meta(df: pl.DataFrame) -> pl.DataFrame:
    if "cluster_id" not in df.columns:
        return df
    _cid = pl.col("cluster_id").cast(pl.Utf8, strict=False).fill_null("").str.strip_chars()
    _year = pl.col("pub_year").cast(pl.Int32, strict=False)
    _has_cid = _cid != ""
    mins = _cluster_min_year(df, "cluster_id", "pub_year")
    merged = df.join(mins, on="cluster_id", how="left")
    return merged.filter(
        (~_has_cid) | pl.col("_cluster_min_pub_year").is_null() | (_year == pl.col("_cluster_min_pub_year"))
    ).drop("_cluster_min_pub_year")


# ──────────────────────────────────────────────────────────────────────────────
# Exposure tables
# ──────────────────────────────────────────────────────────────────────────────


def _exposure_from_work_meta(work_meta_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    _meta_y = (
        work_meta_df.select(
            pl.col("work_id").alias("ppa_work_id"),
            pub_year_i=pl.col("pub_year").cast(pl.Int32, strict=False),
            page_raw=pl.col("page_count"),
        )
        .filter(pl.col("pub_year_i").is_not_null())
        .with_columns(
            missing_pages=pl.col("page_raw").is_null(),
            page_count_filled=pl.col("page_raw").fill_null(0).cast(pl.Int64),
        )
    )
    year_df = (
        _meta_y.group_by("pub_year_i")
        .agg(
            n_works=pl.len(),
            total_pages=pl.col("page_count_filled").sum(),
            n_works_missing_page_count=pl.col("missing_pages").sum(),
        )
        .rename({"pub_year_i": "ppa_pub_year"})
        .sort("ppa_pub_year")
    )
    decade_df = (
        year_df.with_columns(ppa_pub_decade=(pl.col("ppa_pub_year").floordiv(10) * 10).cast(pl.Int32))
        .group_by("ppa_pub_decade")
        .agg(
            n_works=pl.col("n_works").sum(),
            total_pages=pl.col("total_pages").sum(),
            n_works_missing_page_count=pl.col("n_works_missing_page_count").sum(),
        )
        .sort("ppa_pub_decade")
    )
    return year_df, decade_df


# ──────────────────────────────────────────────────────────────────────────────
# Public dataclass + builder
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExcerptUniverse:
    excerpts_df: pl.DataFrame
    excerpts_df_full: pl.DataFrame
    excerpts_df_collapsed: pl.DataFrame
    host_work_meta_df: pl.DataFrame
    work_meta_df_raw: pl.DataFrame
    exposure_year_df: pl.DataFrame
    exposure_decade_df: pl.DataFrame

    collection_filter: tuple[str, ...]
    collection_filter_mode: CollectionFilterMode
    deduplicate_exact_scans: bool
    collapse_clusters: bool
    start_year: int | None
    end_year: int | None
    exclude_poem_ids: tuple[str, ...]
    expo_denom: ExpoDenom

    stage_counts: dict[str, int] = field(default_factory=dict)
    dropped_by_stage: dict[str, int] = field(default_factory=dict)
    collection_breakdown: pl.DataFrame | None = None
    dropped_by_blocklist: pl.DataFrame | None = None

    def ribbon_title_note(self) -> str | None:
        parts: list[str] = []
        if self.collection_filter:
            logic = "all tags" if self.collection_filter_mode == "all" else "any tag"
            parts.append(f"collections: {logic} [{', '.join(self.collection_filter)}]")
        if self.deduplicate_exact_scans:
            parts.append("exact duplicate scans deduplicated")
        if self.collapse_clusters:
            parts.append("host clusters collapsed to earliest dated work")
        if self.start_year is not None or self.end_year is not None:
            lo = str(self.start_year) if self.start_year is not None else "min"
            hi = str(self.end_year) if self.end_year is not None else "max"
            parts.append(f"host year window: [{lo}, {hi}]")
        if self.exclude_poem_ids:
            parts.append(f"{len(self.exclude_poem_ids)} poem_id(s) excluded")
        return " | ".join(parts) if parts else None


def build_excerpt_universe(
    *,
    ppa_work_metadata_path: str | Path,
    passim_excerpts_path: str | Path,
    poem_meta_path: str | Path,
    collection_filter: str | tuple[str, ...] | None = "Literary",
    collection_filter_mode: CollectionFilterMode = "any",
    deduplicate_exact_scans: bool = True,
    collapse_clusters: bool = True,
    start_year: int | None = None,
    end_year: int | None = None,
    exclude_poem_ids: tuple[str, ...] = (),
    expo_denom: ExpoDenom = "total_pages",
) -> ExcerptUniverse:
    """Build PPA-side universe from raw Passim + metadata files."""
    if collection_filter_mode not in ("any", "all"):
        raise ValueError("collection_filter_mode must be 'any' or 'all'")
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    filter_tags = _normalize_collection_filter(collection_filter)

    excerpts_path = Path(passim_excerpts_path)
    work_meta_path = Path(ppa_work_metadata_path)
    pm_path = Path(poem_meta_path)

    work_meta_raw = _load_ppa_work_metadata(work_meta_path)
    excerpts_full = _load_raw_excerpts(excerpts_path, work_meta_path, pm_path)

    stage: dict[str, int] = {
        "raw_excerpts": excerpts_full.height,
        "raw_work_meta": work_meta_raw.height,
        "raw_appearances": n_poem_host_appearances(excerpts_full),
    }
    dropped: dict[str, int] = {}
    excerpts_current = excerpts_full
    work_meta_current = work_meta_raw

    # Stage 1: collection filter
    collection_breakdown = None
    if filter_tags:
        before_ex, before_wm = excerpts_current.height, work_meta_current.height
        before_app = n_poem_host_appearances(excerpts_current)
        excerpts_current = excerpts_current.filter(
            _collection_filter_mask("ppa_collections", filter_tags, collection_filter_mode)
        )
        work_meta_current = work_meta_current.filter(
            _collection_filter_mask("collections", filter_tags, collection_filter_mode)
        )
        dropped["collection_filter_excerpts"] = before_ex - excerpts_current.height
        dropped["collection_filter_work_meta"] = before_wm - work_meta_current.height
        dropped["collection_filter_appearances"] = before_app - n_poem_host_appearances(excerpts_current)
        collection_breakdown = (
            excerpts_current.group_by("ppa_collections").agg(pl.len().alias("n")).sort("n", descending=True)
        )
    stage["after_collection_excerpts"] = excerpts_current.height
    stage["after_collection_work_meta"] = work_meta_current.height
    stage["after_collection_appearances"] = n_poem_host_appearances(excerpts_current)

    # Stage 2: Deduplicate exact scans (Same Title, Same Year, Same Volume)
    # Keep the scan that yielded the most excerpts in the current universe.
    if deduplicate_exact_scans:
        before_ex, before_wm = excerpts_current.height, work_meta_current.height
        before_app = n_poem_host_appearances(excerpts_current)

        # Count excerpts per work
        ex_counts = excerpts_current.group_by("ppa_work_id").agg(pl.len().alias("n_excerpts"))
        
        # Join with work_meta
        wm_eval = work_meta_current.join(
            ex_counts, left_on="work_id", right_on="ppa_work_id", how="left"
        ).with_columns(pl.col("n_excerpts").fill_null(0))
        
        # Sort by excerpt count (descending) so the best scan is first
        wm_eval = wm_eval.sort("n_excerpts", descending=True)
        
        # Drop duplicates based on title, pub_year, volume
        # We fill nulls in volume temporarily to ensure they group correctly if needed,
        # but polars unique handles nulls as equal values automatically.
        work_meta_current = wm_eval.unique(
            subset=["title", "pub_year", "volume"], keep="first", maintain_order=True
        ).drop("n_excerpts")
        
        valid_works = work_meta_current["work_id"]
        excerpts_current = excerpts_current.filter(pl.col("ppa_work_id").is_in(valid_works))
        
        dropped["dedup_exact_excerpts"] = before_ex - excerpts_current.height
        dropped["dedup_exact_work_meta"] = before_wm - work_meta_current.height
        dropped["dedup_exact_appearances"] = before_app - n_poem_host_appearances(excerpts_current)
    stage["after_dedup_exact_excerpts"] = excerpts_current.height
    stage["after_dedup_exact_work_meta"] = work_meta_current.height
    stage["after_dedup_exact_appearances"] = n_poem_host_appearances(excerpts_current)

    # Keep collapsed companion for sensitivity checks
    excerpts_collapsed = _apply_cluster_collapse_excerpts(excerpts_current)

    # Stage 3: cluster collapse
    if collapse_clusters:
        before_ex, before_wm = excerpts_current.height, work_meta_current.height
        before_app = n_poem_host_appearances(excerpts_current)
        excerpts_current = excerpts_collapsed
        work_meta_current = _apply_cluster_collapse_work_meta(work_meta_current)
        dropped["cluster_collapse_excerpts"] = before_ex - excerpts_current.height
        dropped["cluster_collapse_work_meta"] = before_wm - work_meta_current.height
        dropped["cluster_collapse_appearances"] = before_app - n_poem_host_appearances(excerpts_current)
    stage["after_collapse_excerpts"] = excerpts_current.height
    stage["after_collapse_work_meta"] = work_meta_current.height
    stage["after_collapse_appearances"] = n_poem_host_appearances(excerpts_current)

    # Stage 4: optional host publication-year window (after cluster collapse).
    if start_year is not None or end_year is not None:
        before_ex, before_wm = excerpts_current.height, work_meta_current.height
        before_app = n_poem_host_appearances(excerpts_current)
        y_ex = pl.col("ppa_pub_year").cast(pl.Int32, strict=False)
        y_wm = pl.col("pub_year").cast(pl.Int32, strict=False)
        keep_ex = y_ex.is_not_null()
        keep_wm = y_wm.is_not_null()
        if start_year is not None:
            keep_ex = keep_ex & (y_ex >= start_year)
            keep_wm = keep_wm & (y_wm >= start_year)
        if end_year is not None:
            keep_ex = keep_ex & (y_ex <= end_year)
            keep_wm = keep_wm & (y_wm <= end_year)
        excerpts_current = excerpts_current.filter(keep_ex)
        excerpts_collapsed = excerpts_collapsed.filter(keep_ex)
        work_meta_current = work_meta_current.filter(keep_wm)
        dropped["year_window_excerpts"] = before_ex - excerpts_current.height
        dropped["year_window_work_meta"] = before_wm - work_meta_current.height
        dropped["year_window_appearances"] = before_app - n_poem_host_appearances(excerpts_current)
    stage["after_year_window_excerpts"] = excerpts_current.height
    stage["after_year_window_work_meta"] = work_meta_current.height
    stage["after_year_window_appearances"] = n_poem_host_appearances(excerpts_current)

    # Stage 5: poem-id blocklist (excerpt rows only)
    dropped_by_blocklist = None
    if exclude_poem_ids:
        ids = list(exclude_poem_ids)
        before = excerpts_current.height
        before_app = n_poem_host_appearances(excerpts_current)
        dropped_by_blocklist = (
            excerpts_current.filter(pl.col("poem_id").is_in(ids))
            .group_by("poem_id")
            .agg(pl.len().alias("n_dropped"))
            .sort("n_dropped", descending=True)
        )
        excerpts_current = excerpts_current.filter(~pl.col("poem_id").is_in(ids))
        excerpts_collapsed = excerpts_collapsed.filter(~pl.col("poem_id").is_in(ids))
        excerpts_full = excerpts_full.filter(~pl.col("poem_id").is_in(ids))
        dropped["blocklist_excerpts"] = before - excerpts_current.height
        dropped["blocklist_appearances"] = before_app - n_poem_host_appearances(excerpts_current)

    stage["after_blocklist_excerpts"] = excerpts_current.height
    stage["final_excerpts"] = excerpts_current.height
    stage["final_work_meta"] = work_meta_current.height
    stage["final_appearances"] = n_poem_host_appearances(excerpts_current)

    exposure_year, exposure_decade = _exposure_from_work_meta(work_meta_current)

    return ExcerptUniverse(
        excerpts_df=excerpts_current,
        excerpts_df_full=excerpts_full,
        excerpts_df_collapsed=excerpts_collapsed,
        host_work_meta_df=work_meta_current,
        work_meta_df_raw=work_meta_raw,
        exposure_year_df=exposure_year,
        exposure_decade_df=exposure_decade,
        collection_filter=filter_tags,
        collection_filter_mode=collection_filter_mode,
        deduplicate_exact_scans=deduplicate_exact_scans,
        collapse_clusters=collapse_clusters,
        start_year=start_year,
        end_year=end_year,
        exclude_poem_ids=tuple(exclude_poem_ids),
        expo_denom=expo_denom,
        stage_counts=stage,
        dropped_by_stage=dropped,
        collection_breakdown=collection_breakdown,
        dropped_by_blocklist=dropped_by_blocklist,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────────────────────────────────────


def _fmt_pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.2f}%" if whole else "n/a"


def print_excerpt_universe_summary(u: ExcerptUniverse) -> None:
    """Human-readable summary of every universe-building stage."""
    sc = u.stage_counts
    dr = u.dropped_by_stage
    print("Passim excerpt universe\n" + "─" * 72)
    print(f"  raw excerpts loaded          : {sc.get('raw_excerpts', 0):>10,}")
    print(f"  raw host metadata rows       : {sc.get('raw_work_meta', 0):>10,}")
    print(
        f"  distinct poem–host pairs     : {sc.get('raw_appearances', 0):>10,}  "
        f"(unique poem_id ∧ ppa_work_id; breadth / lifetime_works unit)"
    )

    if u.collection_filter:
        logic = "all tags" if u.collection_filter_mode == "all" else "any tag"
        print(
            f"\n[1] Collection filter ({logic}: {', '.join(u.collection_filter)})\n"
            f"    excerpts : {sc['raw_excerpts']:>10,} → {sc['after_collection_excerpts']:>10,}"
            f"  (dropped {dr.get('collection_filter_excerpts', 0):,}, "
            f"{_fmt_pct(dr.get('collection_filter_excerpts', 0), sc['raw_excerpts'])})"
        )
        print(
            f"    appear.  : {sc['raw_appearances']:>10,} → {sc['after_collection_appearances']:>10,}"
            f"  (dropped {dr.get('collection_filter_appearances', 0):,}, "
            f"{_fmt_pct(dr.get('collection_filter_appearances', 0), sc['raw_appearances'])})"
        )
        print(
            f"    host meta: {sc['raw_work_meta']:>10,} → {sc['after_collection_work_meta']:>10,}"
            f"  (dropped {dr.get('collection_filter_work_meta', 0):,}, "
            f"{_fmt_pct(dr.get('collection_filter_work_meta', 0), sc['raw_work_meta'])})"
        )
    else:
        print("\n[1] Collection filter: SKIPPED (collection_filter=None)")

    if u.deduplicate_exact_scans:
        print(
            f"\n[2] Deduplicate exact scans (same title, year, volume)\n"
            f"    excerpts : {sc['after_collection_excerpts']:>10,} → {sc['after_dedup_exact_excerpts']:>10,}"
            f"  (dropped {dr.get('dedup_exact_excerpts', 0):,})"
        )
        print(
            f"    appear.  : {sc['after_collection_appearances']:>10,} → {sc['after_dedup_exact_appearances']:>10,}"
            f"  (dropped {dr.get('dedup_exact_appearances', 0):,})"
        )
        print(
            f"    host meta: {sc['after_collection_work_meta']:>10,} → {sc['after_dedup_exact_work_meta']:>10,}"
            f"  (dropped {dr.get('dedup_exact_work_meta', 0):,})"
        )
    else:
        print("\n[2] Deduplicate exact scans: SKIPPED (deduplicate_exact_scans=False)")

    if u.collapse_clusters:
        print(
            f"\n[3] Cluster collapse (earliest pub_year per cluster_id)\n"
            f"    excerpts : {sc['after_dedup_exact_excerpts']:>10,} → {sc['after_collapse_excerpts']:>10,}"
            f"  (dropped {dr.get('cluster_collapse_excerpts', 0):,})"
        )
        print(
            f"    appear.  : {sc['after_dedup_exact_appearances']:>10,} → {sc['after_collapse_appearances']:>10,}"
            f"  (dropped {dr.get('cluster_collapse_appearances', 0):,})"
        )
        print(
            f"    host meta: {sc['after_dedup_exact_work_meta']:>10,} → {sc['after_collapse_work_meta']:>10,}"
            f"  (dropped {dr.get('cluster_collapse_work_meta', 0):,})"
        )
    else:
        print("\n[3] Cluster collapse: SKIPPED (collapse_clusters=False)")

    if u.start_year is not None or u.end_year is not None:
        lo = str(u.start_year) if u.start_year is not None else "min"
        hi = str(u.end_year) if u.end_year is not None else "max"
        print(
            f"\n[4] Host publication-year window [{lo}, {hi}]\n"
            f"    excerpts : {sc['after_collapse_excerpts']:>10,} → {sc['after_year_window_excerpts']:>10,}"
            f"  (dropped {dr.get('year_window_excerpts', 0):,})"
        )
        print(
            f"    appear.  : {sc['after_collapse_appearances']:>10,} → {sc['after_year_window_appearances']:>10,}"
            f"  (dropped {dr.get('year_window_appearances', 0):,})"
        )
        print(
            f"    host meta: {sc['after_collapse_work_meta']:>10,} → {sc['after_year_window_work_meta']:>10,}"
            f"  (dropped {dr.get('year_window_work_meta', 0):,})"
        )
    else:
        print("\n[4] Host publication-year window: SKIPPED")

    if u.exclude_poem_ids:
        print(
            f"\n[5] Poem-id blocklist ({len(u.exclude_poem_ids)} ids)\n"
            f"    excerpts : {sc['after_year_window_excerpts']:>10,} → {sc['after_blocklist_excerpts']:>10,}"
            f"  (dropped {dr.get('blocklist_excerpts', 0):,})"
        )
        print(
            f"    appear.  : {sc['after_year_window_appearances']:>10,} → {sc['final_appearances']:>10,}"
            f"  (dropped {dr.get('blocklist_appearances', 0):,})"
        )
        if u.dropped_by_blocklist is not None and u.dropped_by_blocklist.height:
            print("    blocked ids with per-id row counts:")
            for row in u.dropped_by_blocklist.iter_rows(named=True):
                print(f"      {str(row['poem_id']):<50s} {row['n_dropped']:>10,}")
    else:
        print("\n[5] Poem-id blocklist: none")

    years_y = u.exposure_year_df["ppa_pub_year"]
    print("\n[6] Exposure tables (from reduced host universe)")
    print(f"    exposure_year_df : {u.exposure_year_df.shape}  years {int(years_y.min())}–{int(years_y.max())}")
    print(
        f"    exposure_decade_df: {u.exposure_decade_df.shape}  "
        f"Σ n_works={int(u.exposure_year_df['n_works'].sum()):,}  "
        f"Σ total_pages={int(u.exposure_year_df['total_pages'].sum()):,}"
    )
    n_missing = int(u.exposure_year_df["n_works_missing_page_count"].sum())
    if n_missing:
        print(f"    host works missing page_count: {n_missing:,}")

    print("\nFinal active universe")
    print(f"  excerpts_df              : {u.excerpts_df.shape}")
    print(f"  distinct poem–host pairs : {sc.get('final_appearances', 0):>10,}")
    print(f"  host_work_meta_df        : {u.host_work_meta_df.shape}")
    print(f"  excerpts_df_full (ref)   : {u.excerpts_df_full.shape}")
    print(f"  excerpts_df_collapsed    : {u.excerpts_df_collapsed.shape}")
    note = u.ribbon_title_note()
    print(f"  ribbon title note        : {note if note else '(no filters active)'}")

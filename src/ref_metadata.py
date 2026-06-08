"""Shared reference-catalogue parsing and ETL for poem_trajectories notebook.

Parsing helpers are used when building `reference_poem_metadata_df` (CH + internet).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import polars as pl

CH_CORPUS = "chadwyck-healey"
INET_CORPUS = "internet_poems"
UNICODE_BOM = "\ufeff"

# Single physical column order for both branches. ``vertical`` concat matches by *position*;
# if CH vs internet ever differ in column order, Polars raises ``schema names differ``.
REFERENCE_METADATA_COLUMNS: tuple[str, ...] = (
    "poem_id",
    "ref_corpus",
    "ref_md_edition_floor_year",
    "ref_md_birth_year_wd",
    "ref_md_death_year_wd",
    "ref_md_ch_birth_lo",
    "ref_md_period",
    "ref_md_catalog_title",
    "ref_md_author_lastname",
    "ref_md_author_firstname",
    "ref_md_qid",
    "ref_md_wikidata_url",
    "ref_md_sex_or_gender_wd",
    "ref_md_country_of_citizenship_wd",
    "ref_md_provenance",
)


def strip_unicode_bom_expr(col: pl.Expr) -> pl.Expr:
    """Strip leading U+FEFF from a Utf8 column (header/value BOM quirks)."""
    return col.cast(pl.Utf8, strict=False).fill_null("").str.strip_chars().str.strip_chars_start(
        UNICODE_BOM
    )


def ch_birth_lo_from_string(s: object) -> int | None:
    """Lower bound year from Chadwyck-style birth string (first plausible 4-digit year)."""
    if s is None:
        return None
    t = str(s).strip()
    if not t or t.lower() == "nan":
        return None
    nums = [int(m) for m in re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", t)]
    return min(nums) if nums else None


def wd_year_from_iso_col(raw: pl.Expr) -> pl.Expr:
    """Parse a signed year from Wikidata-style date strings (best-effort).

    Handles all three formats that actually appear in the catalogs:

    - ISO with dashes     : ``"-0069-10-13"``, ``"1564-04-23"`` → ``-69``, ``1564``
    - 4-digit bare year   : ``"-1850"``, ``"1850"``           → ``-1850``, ``1850``
    - 1- to 4-digit bare  : ``"-70"``, ``"70"``               → ``-70``, ``70``

    Earlier versions of this parser short-circuited on bare negative years
    (e.g. ``"-70"`` → ``null``), which dropped the internet-poems Virgil
    entry and any Chadwyck classical author whose string wasn't full ISO.
    The new implementation extracts the absolute year first (from either
    ISO or bare form) and applies the sign at the end, so no valid
    representation is silently lost.
    """
    r = raw.cast(pl.Utf8).str.strip_chars().fill_null("")
    # ISO form: optional leading "-", 4-digit year, followed by a "-" delimiter
    # (the trailing dash is consumed but not captured — polars' regex crate does
    # not support lookaheads).
    y_iso = r.str.extract(r"^-?(\d{4})-", 1).cast(pl.Int32, strict=False)
    # Bare form: optional leading "-", 1-4 digits, end of string.
    y_bare = r.str.extract(r"^-?(\d{1,4})$", 1).cast(pl.Int32, strict=False)
    abs_year = pl.coalesce([y_iso, y_bare])
    is_neg = r.str.starts_with("-")
    return pl.when(is_neg).then(-abs_year).otherwise(abs_year)


def _fold_ch_meta_by_canonical(
    ch_meta: pl.DataFrame, canonical_map: Mapping[str, str]
) -> pl.DataFrame:
    if not canonical_map:
        return ch_meta
    _map_tbl = pl.DataFrame(
        {
            "poem_id": list(canonical_map.keys()),
            "poem_id_canonical": list(canonical_map.values()),
        }
    )
    return (
        ch_meta.join(_map_tbl, on="poem_id", how="left")
        .with_columns(pl.coalesce(["poem_id_canonical", "poem_id"]).alias("_poem_id_join"))
        .drop("poem_id_canonical")
        .group_by("_poem_id_join")
        .agg(
            pl.col("ref_md_edition_floor_year").min().alias("ref_md_edition_floor_year"),
            pl.col("ref_md_birth_year_wd").min().alias("ref_md_birth_year_wd"),
            pl.col("ref_md_death_year_wd").min().alias("ref_md_death_year_wd"),
            pl.col("ref_md_ch_birth_lo").min().alias("ref_md_ch_birth_lo"),
            pl.col("ref_md_period").drop_nulls().first().alias("ref_md_period"),
            pl.col("ref_md_catalog_title").drop_nulls().first().alias("ref_md_catalog_title"),
            pl.col("ref_md_author_lastname").drop_nulls().first().alias("ref_md_author_lastname"),
            pl.col("ref_md_author_firstname").drop_nulls().first().alias("ref_md_author_firstname"),
            pl.col("ref_md_qid").drop_nulls().first().alias("ref_md_qid"),
            pl.col("ref_md_wikidata_url").drop_nulls().first().alias("ref_md_wikidata_url"),
            pl.col("ref_md_sex_or_gender_wd").drop_nulls().first().alias("ref_md_sex_or_gender_wd"),
            pl.col("ref_md_country_of_citizenship_wd")
            .drop_nulls()
            .first()
            .alias("ref_md_country_of_citizenship_wd"),
            pl.col("ref_corpus").drop_nulls().first().alias("ref_corpus"),
            pl.col("ref_md_provenance").drop_nulls().first().alias("ref_md_provenance"),
        )
        .rename({"_poem_id_join": "poem_id"})
    )


def build_ch_reference_branch(repo_root: Path, canonical_map: Mapping[str, str] | None) -> pl.DataFrame:
    _pm_path = repo_root / "data" / "poetry_metadata.csv"
    if not _pm_path.exists():
        raise FileNotFoundError(f"Missing {_pm_path}")
    _pm = pl.read_csv(
        _pm_path,
        infer_schema_length=50000,
        schema_overrides={
            "author_birth_origch": pl.Utf8,
            "author_death_origch": pl.Utf8,
            "author_period_origch": pl.Utf8,
            "transl_birth_origch": pl.Utf8,
            "transl_death_origch": pl.Utf8,
            "genre": pl.Utf8,
            "rhymes": pl.Utf8,
        },
    )
    _tid = pl.col("title_id").cast(pl.Utf8).str.strip_chars()
    _pm = _pm.filter(_tid.is_not_null() & (_tid != ""))
    _year_e = pl.col("edition_year").cast(pl.Int32, strict=False)
    _pm_sorted = _pm.with_columns(_year_e.alias("_ey")).sort("_ey", nulls_last=True)

    ch_meta = (
        _pm_sorted.group_by(_tid.alias("poem_id"))
        .agg(
            pl.col("_ey").min().alias("ref_md_edition_floor_year"),
            pl.col("date_of_birth_wd").drop_nulls().first().alias("_dob_wd_raw"),
            pl.col("date_of_death_wd").drop_nulls().first().alias("_dod_wd_raw"),
            pl.col("author_birth_origch").drop_nulls().first().alias("_ch_birth_raw"),
            pl.col("period").drop_nulls().first().alias("ref_md_period"),
            pl.col("title_main_origch").drop_nulls().first().alias("ref_md_catalog_title"),
            pl.col("author_lastname").drop_nulls().first().alias("ref_md_author_lastname"),
            pl.col("author_firstname").drop_nulls().first().alias("ref_md_author_firstname"),
            pl.col("qid").drop_nulls().first().alias("ref_md_qid"),
            pl.col("wikidata_url").drop_nulls().first().alias("ref_md_wikidata_url"),
            pl.col("sex_or_gender_wd").drop_nulls().first().alias("ref_md_sex_or_gender_wd"),
            pl.col("country_of_citizenship_wd")
            .drop_nulls()
            .first()
            .alias("ref_md_country_of_citizenship_wd"),
        )
        .with_columns(
            wd_year_from_iso_col(pl.col("_dob_wd_raw")).alias("ref_md_birth_year_wd"),
            wd_year_from_iso_col(pl.col("_dod_wd_raw")).alias("ref_md_death_year_wd"),
            pl.col("_ch_birth_raw")
            .map_elements(ch_birth_lo_from_string, return_dtype=pl.Int32)
            .alias("ref_md_ch_birth_lo"),
            pl.lit(CH_CORPUS).alias("ref_corpus"),
            pl.lit("poetry_metadata.csv").alias("ref_md_provenance"),
        )
        .drop(["_dob_wd_raw", "_dod_wd_raw", "_ch_birth_raw"])
    )
    return _fold_ch_meta_by_canonical(ch_meta, canonical_map or {})


def _harmonize_reference_branch_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Cast CH and internet branches to identical dtypes and **column order**."""
    out = df.select(
        strip_unicode_bom_expr(pl.col("poem_id")).alias("poem_id"),
        pl.col("ref_corpus").cast(pl.Utf8, strict=False).str.strip_chars(),
        pl.col("ref_md_edition_floor_year").cast(pl.Int32, strict=False),
        pl.col("ref_md_birth_year_wd").cast(pl.Int32, strict=False),
        pl.col("ref_md_death_year_wd").cast(pl.Int32, strict=False),
        pl.col("ref_md_ch_birth_lo").cast(pl.Int32, strict=False),
        pl.col("ref_md_period").cast(pl.Utf8, strict=False),
        pl.col("ref_md_catalog_title").cast(pl.Utf8, strict=False),
        pl.col("ref_md_author_lastname").cast(pl.Utf8, strict=False),
        pl.col("ref_md_author_firstname").cast(pl.Utf8, strict=False),
        pl.col("ref_md_qid").cast(pl.Utf8, strict=False),
        pl.col("ref_md_wikidata_url").cast(pl.Utf8, strict=False),
        pl.col("ref_md_sex_or_gender_wd").cast(pl.Utf8, strict=False),
        pl.col("ref_md_country_of_citizenship_wd").cast(pl.Utf8, strict=False),
        pl.col("ref_md_provenance").cast(pl.Utf8, strict=False),
    )
    return out.select(REFERENCE_METADATA_COLUMNS)


def build_internet_reference_branch(repo_root: Path) -> pl.DataFrame:
    _inet_path = repo_root / "data/internet_poems/internet_poems_metadata_enriched.csv"
    if not _inet_path.exists():
        raise FileNotFoundError(f"Missing {_inet_path}")
    _inet = pl.read_csv(_inet_path)
    return (
        _inet.with_columns(
            pl.col("filename")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.replace(r"\.txt$", "")
            .alias("poem_id"),
            pl.lit(INET_CORPUS).alias("ref_corpus"),
            pl.lit("internet_poems_metadata_enriched.csv").alias("ref_md_provenance"),
            pl.lit(None, dtype=pl.Int32).alias("ref_md_edition_floor_year"),
            pl.lit(None, dtype=pl.Int32).alias("ref_md_ch_birth_lo"),
        )
        .with_columns(
            wd_year_from_iso_col(pl.col("date_of_birth_wd")).alias("ref_md_birth_year_wd"),
            wd_year_from_iso_col(pl.col("date_of_death_wd")).alias("ref_md_death_year_wd"),
        )
        .select(
            [
                "poem_id",
                "ref_corpus",
                "ref_md_edition_floor_year",
                "ref_md_birth_year_wd",
                "ref_md_death_year_wd",
                "ref_md_ch_birth_lo",
                pl.col("period").alias("ref_md_period"),
                pl.col("title").alias("ref_md_catalog_title"),
                pl.col("author_lastname").alias("ref_md_author_lastname"),
                pl.col("author_firstname").alias("ref_md_author_firstname"),
                pl.col("qid").alias("ref_md_qid"),
                pl.col("wikidata_url").alias("ref_md_wikidata_url"),
                pl.col("sex_or_gender_wd").alias("ref_md_sex_or_gender_wd"),
                pl.col("country_of_citizenship_wd").alias("ref_md_country_of_citizenship_wd"),
                "ref_md_provenance",
            ]
        )
    )


def build_reference_poem_metadata_df(
    repo_root: Path, *, canonical_map: Mapping[str, str] | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Stack CH + internet reference rows (one row per ``(poem_id, ref_corpus)``).

    Returns ``(reference_poem_metadata_df, poetry_meta_poem_df)`` where the second
    frame is the CH-only slice (same name as legacy notebook variables).
    """
    _ch_meta = _harmonize_reference_branch_schema(build_ch_reference_branch(repo_root, canonical_map))
    _inet_meta = _harmonize_reference_branch_schema(build_internet_reference_branch(repo_root))
    # ``diagonal`` aligns by **column name** (union); then enforce one physical order for joins downstream.
    reference_poem_metadata_df = pl.concat(
        [_ch_meta, _inet_meta],
        how="diagonal",
    ).select(REFERENCE_METADATA_COLUMNS)
    _dup = reference_poem_metadata_df.group_by(["poem_id", "ref_corpus"]).len().filter(pl.col("len") > 1)
    if _dup.height:
        raise ValueError(f"Duplicate (poem_id, ref_corpus) in reference_poem_metadata_df: {_dup}")
    poetry_meta_poem_df = reference_poem_metadata_df.filter(pl.col("ref_corpus") == CH_CORPUS)
    return reference_poem_metadata_df, poetry_meta_poem_df


def join_reference_metadata_onto_excerpts(
    excerpts_df: pl.DataFrame, reference_poem_metadata_df: pl.DataFrame
) -> pl.DataFrame:
    """Left-join catalogue fields onto excerpt rows without row multiplication."""
    return excerpts_df.join(
        reference_poem_metadata_df,
        on=["poem_id", "ref_corpus"],
        how="left",
    )

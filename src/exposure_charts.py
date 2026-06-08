"""Altair helpers: corpus exposure ribbons by PPA host publication year or decade.

Pure chart code. The exposure frames themselves (``exposure_year_df`` /
``exposure_decade_df``) are built by :mod:`excerpt_universe` as part of
``build_excerpt_universe``; this module only renders them.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import altair as alt
import pandas as pd
import polars as pl

ExpoDenom = Literal["total_pages", "n_works"]

# Default horizontal guide for ``total_pages`` ribbons (pages per year / decade).
_DEFAULT_REFERENCE_PAGES = 15_000

# Human-readable labels for the exposure denominator — used in chart titles.
_DENOM_LABEL: dict[str, str] = {
    "total_pages": "Total pages",
    "n_works": "Total works",
}


def _denom_label(expo_denom: str) -> str:
    return _DENOM_LABEL.get(expo_denom, expo_denom)


def _format_pages_label(y: float) -> str:
    return f"{int(round(y)):,} pages"


def _year_ribbon_title(expo_denom: ExpoDenom) -> str:
    """Short ribbon subtitle (host publication year on the x-axis)."""
    if expo_denom == "total_pages":
        return "Total host-work pages per year"
    if expo_denom == "n_works":
        return "Host works per publication year"
    return f"Corpus exposure — {_denom_label(expo_denom)}"


def _decade_ribbon_title(expo_denom: ExpoDenom) -> str:
    if expo_denom == "total_pages":
        return "Total host-work pages per decade"
    if expo_denom == "n_works":
        return "Host works per publication decade"
    return f"Corpus exposure by decade — {_denom_label(expo_denom)}"


def _year_domain(df: pl.DataFrame, col: str) -> list[int]:
    """Decade-rounded [min, max] from an integer year/decade column.

    Floors the min down to the enclosing decade and ceils the max up to the next
    decade boundary so the axis ends cleanly without Altair auto-padding another
    decade or two of empty whitespace beyond the data.
    """
    series = df[col].cast(pl.Int32, strict=False).drop_nulls()
    lo, hi = int(series.min()), int(series.max())
    return [(lo // 10) * 10, ((hi + 9) // 10) * 10]


def pad_exposure_year_years(
    exposure_year_df: pl.DataFrame,
    year_lo: int,
    year_hi: int,
) -> pl.DataFrame:
    """One row per integer ``ppa_pub_year`` in ``[year_lo, year_hi]`` (inclusive).

    Host metadata can omit years that still appear in modelled rate panels; without
    dense years the last ribbon ``mark_bar`` can sit short of the shared x-domain
    end. Missing years are filled with zero exposure counts.
    """
    lo, hi = int(year_lo), int(year_hi)
    if lo > hi:
        lo, hi = hi, lo
    grid = pl.DataFrame({"ppa_pub_year": list(range(lo, hi + 1))}).with_columns(
        pl.col("ppa_pub_year").cast(pl.Int32)
    )
    if exposure_year_df.height == 0:
        return grid.with_columns(
            pl.lit(0).cast(pl.Int64).alias("n_works"),
            pl.lit(0).cast(pl.Int64).alias("total_pages"),
            pl.lit(0).cast(pl.Int64).alias("n_works_missing_page_count"),
        )
    out = grid.join(
        exposure_year_df.with_columns(pl.col("ppa_pub_year").cast(pl.Int32, strict=False)),
        on="ppa_pub_year",
        how="left",
    )
    for name in out.columns:
        if name == "ppa_pub_year":
            continue
        out = out.with_columns(pl.col(name).fill_null(0))
    return out


def exposure_ribbon_year(
    exposure_year_df: pl.DataFrame,
    *,
    expo_denom: ExpoDenom = "total_pages",
    width: int = 720,
    height: int = 62,
    title: Optional[str] = None,
    show_reference_line: bool = True,
    reference_y: Optional[float] = None,
    reference_label: Optional[str] = None,
    x_domain: Optional[Tuple[int, int]] = None,
) -> alt.Chart:
    """Ribbon of exposure by host publication year.

    For ``expo_denom == "total_pages"``, by default draws a dashed guide at
    :data:`_DEFAULT_REFERENCE_PAGES` (15,000) with label "15,000 pages".
    Pass ``reference_y`` to override; set ``show_reference_line=False`` to hide.
    For ``n_works``, a line is drawn only if ``reference_y`` is not ``None``.

    If ``x_domain`` is ``(lo, hi)``, the x-scale uses those integer years exactly
    (``nice=False``) instead of decade-rounded bounds from the data. Use this so
    the ribbon aligns with a main panel that starts at e.g. ``MODEL_YEAR_MIN``
    rather than a decade floor below the first year.
    """
    if x_domain is not None:
        lo, hi = int(x_domain[0]), int(x_domain[1])
        if lo > hi:
            lo, hi = hi, lo
        _domain = [lo, hi]
        ey_pl = pad_exposure_year_years(exposure_year_df, lo, hi)
    else:
        _domain = _year_domain(exposure_year_df, "ppa_pub_year")
        ey_pl = exposure_year_df
    _ed = ey_pl.to_pandas()
    y_field = expo_denom
    # Quantitative Scale in Vega-Lite has no ``clip``; use ``nice=False`` + exact domain only.
    _x_scale_kw = dict(domain=_domain, nice=False)
    _x_axis = alt.Axis(format=".0f", title=None, labels=False, ticks=False)

    # Same mark as the decade-free ribbon: light gray bars from y=0 (not ``mark_rect``, which can
    # read as inverted cut-outs with some themes / shared-scale rendering). Dense years from
    # ``pad_exposure_year_years`` when ``x_domain`` is set keep the last model year (e.g. 1920) on-bar.
    x_enc = alt.X(
        "ppa_pub_year:Q",
        scale=alt.Scale(**_x_scale_kw),
        axis=_x_axis,
    )
    y_enc = alt.Y(f"{y_field}:Q", axis=None, title=None)
    bars = (
        alt.Chart(_ed)
        .mark_bar(color="#d3d3d3")
        .encode(
            x_enc,
            y_enc,
            tooltip=["ppa_pub_year", "n_works", "total_pages", "n_works_missing_page_count"],
        )
    )

    layers: list[alt.Chart] = [bars]

    ref_y: Optional[float] = None
    ref_lbl: Optional[str] = None
    if show_reference_line:
        if expo_denom == "total_pages":
            ref_y = float(reference_y if reference_y is not None else _DEFAULT_REFERENCE_PAGES)
            ref_lbl = reference_label or _format_pages_label(ref_y)
        elif expo_denom == "n_works" and reference_y is not None:
            ref_y = float(reference_y)
            ref_lbl = reference_label or f"{int(round(ref_y)):,} works"

    if ref_y is not None:
        _rule_df = pd.DataFrame({y_field: [ref_y]})
        ref_rule = (
            alt.Chart(_rule_df)
            .mark_rule(color="#888888", strokeDash=[4, 4], strokeWidth=1, opacity=0.9)
            .encode(y=alt.Y(f"{y_field}:Q"))
        )
        layers.append(ref_rule)

        _x_left = float(_domain[0])
        _label_df = pd.DataFrame(
            {
                "ppa_pub_year": [_x_left],
                y_field: [ref_y],
                "_lbl": [ref_lbl],
            }
        )
        ref_text = (
            alt.Chart(_label_df)
            .mark_text(align="left", baseline="bottom", dx=4, dy=-3, fontSize=9, color="#555555")
            .encode(
                x=alt.X(
                    "ppa_pub_year:Q",
                    scale=alt.Scale(**_x_scale_kw),
                ),
                y=alt.Y(f"{y_field}:Q"),
                text=alt.Text("_lbl:N"),
            )
        )
        layers.append(ref_text)

    ch = alt.layer(*layers).properties(width=width, height=height)
    if title:
        ch = ch.properties(title=title)
    return ch


def exposure_ribbon_decade(
    exposure_decade_df: pl.DataFrame,
    *,
    expo_denom: ExpoDenom = "total_pages",
    width: int = 700,
    height: int = 62,
    title: Optional[str] = None,
    show_reference_line: bool = True,
    reference_y: Optional[float] = None,
    reference_label: Optional[str] = None,
) -> alt.Chart:
    _ed = exposure_decade_df.to_pandas()
    _domain = _year_domain(exposure_decade_df, "ppa_pub_decade")
    y_field = expo_denom
    x_enc = alt.X(
        "ppa_pub_decade:Q",
        scale=alt.Scale(domain=_domain, nice=False),
        axis=alt.Axis(format=".0f", title=None, labels=False, ticks=False),
    )
    y_enc = alt.Y(f"{y_field}:Q", axis=None, title=None)

    bars = (
        alt.Chart(_ed)
        .mark_bar(color="#d3d3d3", width=8)
        .encode(
            x_enc,
            y_enc,
            tooltip=[
                "ppa_pub_decade",
                "n_works",
                "total_pages",
                "n_works_missing_page_count",
            ],
        )
    )

    layers: list[alt.Chart] = [bars]

    ref_y: Optional[float] = None
    ref_lbl: Optional[str] = None
    if show_reference_line:
        if expo_denom == "total_pages":
            ref_y = float(reference_y if reference_y is not None else _DEFAULT_REFERENCE_PAGES)
            ref_lbl = reference_label or _format_pages_label(ref_y)
        elif expo_denom == "n_works" and reference_y is not None:
            ref_y = float(reference_y)
            ref_lbl = reference_label or f"{int(round(ref_y)):,} works"

    if ref_y is not None:
        _rule_df = pd.DataFrame({y_field: [ref_y]})
        ref_rule = (
            alt.Chart(_rule_df)
            .mark_rule(color="#888888", strokeDash=[4, 4], strokeWidth=1, opacity=0.9)
            .encode(y=alt.Y(f"{y_field}:Q"))
        )
        layers.append(ref_rule)

        _x_left = float(_domain[0])
        _label_df = pd.DataFrame(
            {
                "ppa_pub_decade": [_x_left],
                y_field: [ref_y],
                "_lbl": [ref_lbl],
            }
        )
        ref_text = (
            alt.Chart(_label_df)
            .mark_text(align="left", baseline="bottom", dx=4, dy=-3, fontSize=9, color="#555555")
            .encode(
                x=alt.X(
                    "ppa_pub_decade:Q",
                    scale=alt.Scale(domain=_domain, nice=False),
                ),
                y=alt.Y(f"{y_field}:Q"),
                text=alt.Text("_lbl:N"),
            )
        )
        layers.append(ref_text)

    ch = alt.layer(*layers).properties(width=width, height=height)
    if title:
        ch = ch.properties(title=title)
    return ch


def v_with_year_ribbon(
    main: alt.Chart,
    exposure_year_df: pl.DataFrame,
    *,
    expo_denom: ExpoDenom = "total_pages",
    width: int = 720,
    ribbon_height: int = 62,
    ribbon_show_reference_line: bool = True,
    ribbon_reference_y: Optional[float] = None,
    ribbon_reference_label: Optional[str] = None,
    x_domain: Optional[Tuple[int, int]] = None,
) -> alt.Chart:
    _title = _year_ribbon_title(expo_denom)
    rib = exposure_ribbon_year(
        exposure_year_df,
        expo_denom=expo_denom,
        width=width,
        height=ribbon_height,
        show_reference_line=ribbon_show_reference_line,
        reference_y=ribbon_reference_y,
        reference_label=ribbon_reference_label,
        x_domain=x_domain,
    ).properties(
        title=alt.TitleParams(text=_title, fontSize=11, color="#666666")
    )
    return alt.vconcat(main, rib).resolve_scale(x="shared")


def v_with_decade_ribbon(
    main: alt.Chart,
    exposure_decade_df: pl.DataFrame,
    *,
    expo_denom: ExpoDenom = "total_pages",
    width: int = 700,
    ribbon_height: int = 62,
    ribbon_show_reference_line: bool = True,
    ribbon_reference_y: Optional[float] = None,
    ribbon_reference_label: Optional[str] = None,
) -> alt.Chart:
    _title = _decade_ribbon_title(expo_denom)
    rib = exposure_ribbon_decade(
        exposure_decade_df,
        expo_denom=expo_denom,
        width=width,
        height=ribbon_height,
        show_reference_line=ribbon_show_reference_line,
        reference_y=ribbon_reference_y,
        reference_label=ribbon_reference_label,
    ).properties(
        title=alt.TitleParams(text=_title, fontSize=11, color="#666666")
    )
    return alt.vconcat(main, rib).resolve_scale(x="shared")

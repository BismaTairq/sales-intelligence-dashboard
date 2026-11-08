"""Plotly chart builders. Each function takes a DataFrame and returns a Figure.

Visual rules (from a validated data-viz palette):
- one hue for single-series charts; categorical colours are assigned per entity in a fixed order
  (first three slots only, which validate for colour-blind readers even when all appear together);
- one y-axis per chart, thin marks, recessive grid, ink colours for text (never the series colour);
- every mark has a hover tooltip with the exact value.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (categorical slots 1-3)
BLUE, ORANGE = SERIES[0], SERIES[1]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

LABELS = {"Sales": "Revenue", "Profit": "Profit"}


def _style(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_2),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(family=FONT, color=INK, size=12)),
        bargap=0.35,
    )
    # automargin makes Plotly reserve room for tick labels and axis titles instead of clipping them
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickfont=dict(color=MUTED), ticks="", automargin=True)
    fig.update_yaxes(
        gridcolor=GRID, zeroline=True, zerolinecolor=AXIS, tickfont=dict(color=MUTED), showline=False, ticks="",
        automargin=True,
    )
    return fig


def _round_bars(fig: go.Figure) -> go.Figure:
    try:  # rounded data-ends need a recent Plotly; skip quietly on older versions
        fig.update_traces(marker_cornerradius=4)
    except Exception:
        pass
    return fig


def _empty(message: str = "No data for this selection") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height=240)


def revenue_trend(trend: pd.DataFrame, anomalies: pd.DataFrame | None = None) -> go.Figure:
    if trend.empty:
        return _empty()
    hover = [
        f"<b>{d:%b %Y}</b><br>Revenue ${s:,.0f}"
        + (f"<br>vs previous month {g:+.1f}%" if pd.notna(g) else "")
        for d, s, g in zip(trend["Order Date"], trend["Sales"], trend["MoM Growth %"])
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trend["Order Date"], y=trend["Sales"], mode="lines", name="Revenue",
            line=dict(color=BLUE, width=2), hovertext=hover, hoverinfo="text",
        )
    )
    has_anoms = anomalies is not None and not anomalies.empty
    if has_anoms:
        a_hover = [
            f"<b>{d:%b %Y}</b><br>Revenue ${s:,.0f}<br>Growth {g:+.1f}% (outlier)"
            for d, s, g in zip(anomalies["Order Date"], anomalies["Sales"], anomalies["MoM Growth %"])
        ]
        fig.add_trace(
            go.Scatter(
                x=anomalies["Order Date"], y=anomalies["Sales"], mode="markers", name="Unusual month",
                marker=dict(color=ORANGE, size=10, line=dict(color=SURFACE, width=2)),
                hovertext=a_hover, hoverinfo="text",
            )
        )
    _style(fig, height=340, legend=has_anoms)
    fig.update_xaxes(tickformat="%b %Y")
    fig.update_yaxes(tickprefix="$", tickformat="~s", rangemode="tozero")
    return fig


def profit_bars(trend: pd.DataFrame) -> go.Figure:
    if trend.empty:
        return _empty()
    hover = [f"<b>{d:%b %Y}</b><br>Profit ${p:,.0f}" for d, p in zip(trend["Order Date"], trend["Profit"])]
    fig = go.Figure(go.Bar(x=trend["Order Date"], y=trend["Profit"], marker_color=BLUE, hovertext=hover, hoverinfo="text"))
    _style(fig, height=300)
    fig.update_xaxes(tickformat="%b %Y")
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return _round_bars(fig)


def margin_by_discount(disc: pd.DataFrame) -> go.Figure:
    if disc.empty:
        return _empty()
    hover = [
        f"<b>Discount {b}</b><br>Margin {m:.1f}%<br>{int(n):,} order lines"
        for b, m, n in zip(disc["Bucket"], disc["Margin %"], disc["Lines"])
    ]
    fig = go.Figure(go.Bar(x=disc["Bucket"], y=disc["Margin %"], marker_color=BLUE, hovertext=hover, hoverinfo="text"))
    _style(fig, height=300)
    fig.update_xaxes(title=dict(text="Discount on the order line", font=dict(color=MUTED), standoff=10))
    fig.update_yaxes(ticksuffix="%", tickformat=".0f")
    return _round_bars(fig)


def top_products(top: pd.DataFrame, value: str = "Sales") -> go.Figure:
    if top.empty:
        return _empty()
    d = top.sort_values(value, ascending=True)
    names = d["Product Name"].tolist()
    labels = [n if len(n) <= 34 else n[:33] + "…" for n in names]
    if len(set(labels)) < len(labels):  # truncation made two names identical -> keep full names
        labels = names
    hover = [
        f"<b>{n}</b><br>{LABELS[value]} ${v:,.0f}<br>Margin {m:.1f}%"
        for n, v, m in zip(names, d[value], d["Margin %"])
    ]
    fig = go.Figure(
        go.Bar(x=d[value], y=labels, orientation="h", marker_color=BLUE, hovertext=hover, hoverinfo="text")
    )
    _style(fig, height=max(300, 34 * len(d) + 40))
    fig.update_xaxes(tickprefix="$", tickformat="~s", showgrid=True, gridcolor=GRID, linecolor="rgba(0,0,0,0)")
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(color=INK_2))
    return _round_bars(fig)


def subcategory_profit(sub: pd.DataFrame) -> go.Figure:
    if sub.empty:
        return _empty()
    d = sub.sort_values("Profit", ascending=True)
    hover = [
        f"<b>{n}</b><br>Profit ${p:,.0f}<br>Revenue ${s:,.0f}<br>Margin {m:.1f}%"
        for n, p, s, m in zip(d["Sub-Category"], d["Profit"], d["Sales"], d["Margin %"])
    ]
    fig = go.Figure(
        go.Bar(x=d["Profit"], y=d["Sub-Category"], orientation="h", marker_color=BLUE, hovertext=hover, hoverinfo="text")
    )
    _style(fig, height=max(300, 30 * len(d) + 40))
    fig.update_xaxes(tickprefix="$", tickformat="~s", showgrid=True, gridcolor=GRID, linecolor="rgba(0,0,0,0)")
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(color=INK_2))
    return _round_bars(fig)


def entity_bar(bd: pd.DataFrame, col: str, color_map: dict | None = None) -> go.Figure:
    """Revenue by region / category. `color_map` keeps each entity's colour stable across filters."""
    if bd.empty:
        return _empty()
    colors = [color_map.get(v, BLUE) for v in bd[col]] if color_map else BLUE
    hover = [
        f"<b>{n}</b><br>Revenue ${s:,.0f}<br>Profit ${p:,.0f}<br>Margin {m:.1f}%"
        for n, s, p, m in zip(bd[col], bd["Sales"], bd["Profit"], bd["Margin %"])
    ]
    fig = go.Figure(go.Bar(x=bd[col], y=bd["Sales"], marker_color=colors, hovertext=hover, hoverinfo="text"))
    _style(fig, height=300)
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return _round_bars(fig)

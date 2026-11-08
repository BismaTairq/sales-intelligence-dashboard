"""Sales Intelligence Dashboard (Streamlit).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import inspect
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from generate_sample_data import generate
from src import charts as ch
from src import metrics as m
from src.data_loader import load_data
from src.insights import generate_insights, ungrounded_numbers

ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "data" / "sample_sales.csv"
PLOT_CONFIG = {"displayModeBar": False}

st.set_page_config(page_title="Sales Intelligence Dashboard", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.block-container { padding-top: 3.5rem; max-width: 1400px; }
footer { visibility: hidden; }
.app-title { font-size: 1.7rem; font-weight: 650; margin: 0; color: #0b0b0b; }
.app-sub { color: #52514e; margin: .15rem 0 1.1rem 0; }
.kpi { background: #fcfcfb; border: 1px solid rgba(11,11,11,.10); border-radius: 12px; padding: 14px 16px; }
.kpi .label { font-size: .75rem; color: #52514e; text-transform: uppercase; letter-spacing: .05em; }
.kpi .value { font-size: 1.75rem; font-weight: 650; color: #0b0b0b; line-height: 1.3; }
.kpi .delta { font-size: .8rem; font-weight: 600; }
.kpi .delta.up { color: #006300; }
.kpi .delta.down { color: #d03b3b; }
.kpi .delta.flat { color: #898781; font-weight: 500; }
.kpi .note { color: #898781; font-weight: 400; }
.chart-title { font-weight: 600; color: #0b0b0b; margin: 0; font-size: 1rem; }
.chart-sub { color: #52514e; font-size: .82rem; margin: 0 0 .4rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ small helpers
def _stretch(fn) -> dict:
    """Streamlit renamed use_container_width -> width='stretch'; support both."""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


def show(fig) -> None:
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG, **_stretch(st.plotly_chart))


def panel(title: str, subtitle: str):
    box = st.container(border=True)
    box.markdown(f'<p class="chart-title">{title}</p><p class="chart-sub">{subtitle}</p>', unsafe_allow_html=True)
    return box


def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:.1f}K"
    return f"{sign}${x:,.0f}"


def kpi_card(label: str, value: str, delta: float | None, unit: str = "%") -> str:
    if delta is None:
        d = '<span class="delta flat">– no prior period</span>'
    else:
        up = delta >= 0
        arrow = "▲" if up else "▼"
        d = (
            f'<span class="delta {"up" if up else "down"}">{arrow} {abs(delta):.1f}{unit} '
            f'<span class="note">vs previous period</span></span>'
        )
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>{d}</div>'


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    if not SAMPLE_PATH.exists():
        generate(SAMPLE_PATH)
    return load_data(SAMPLE_PATH)


@st.cache_data(show_spinner=False)
def load_upload(raw: bytes) -> pd.DataFrame:
    return load_data(io.BytesIO(raw))


# ------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Data")
    upload = st.file_uploader("Upload a Superstore-style CSV", type="csv")
    try:
        df_all = load_upload(upload.getvalue()) if upload else load_sample()
    except ValueError as err:
        st.error(str(err))
        st.stop()
    if upload is None:
        st.caption("Showing generated sample data (synthetic, for demo only).")

    st.markdown("### Filters")
    min_d, max_d = df_all["Order Date"].min().date(), df_all["Order Date"].max().date()
    date_range = st.date_input("Order date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    all_regions = sorted(df_all["Region"].unique())
    all_cats = sorted(df_all["Category"].unique())
    regions = st.multiselect("Region", all_regions, default=all_regions)
    categories = st.multiselect("Category", all_cats, default=all_cats)

if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
    st.info("Pick an end date to complete the range.")
    st.stop()

# ------------------------------------------------------------------- filter data
start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
base = df_all[df_all["Region"].isin(regions) & df_all["Category"].isin(categories)]
df = base[(base["Order Date"] >= start) & (base["Order Date"] < end + pd.Timedelta(days=1))]
prev = m.previous_period(base, start, end)

st.markdown('<p class="app-title">Sales Intelligence Dashboard</p>', unsafe_allow_html=True)
if df.empty:
    st.warning("No orders match these filters. Widen the date range or select more regions/categories.")
    st.stop()
st.markdown(
    f'<p class="app-sub">{len(df):,} order lines · {start:%d %b %Y} – {end:%d %b %Y}</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------- KPIs
k, kp = m.compute_kpis(df), m.compute_kpis(prev)
has_prev = not prev.empty
cards = [
    ("Revenue", fmt_money(k["revenue"]), m.pct_change(k["revenue"], kp["revenue"]), "%"),
    ("Profit", fmt_money(k["profit"]), m.pct_change(k["profit"], kp["profit"]), "%"),
    ("Orders", f"{k['orders']:,}", m.pct_change(k["orders"], kp["orders"]), "%"),
    ("Avg order value", fmt_money(k["aov"]), m.pct_change(k["aov"], kp["aov"]), "%"),
    ("Profit margin", f"{k['margin_pct']:.1f}%", (k["margin_pct"] - kp["margin_pct"]) if has_prev else None, " pp"),
]
for col, (label, value, delta, unit) in zip(st.columns(5), cards):
    col.markdown(kpi_card(label, value, delta, unit), unsafe_allow_html=True)
st.write("")

# --------------------------------------------------------------------------- tabs
trend = m.monthly_trend(df)
anomalies = m.detect_anomalies(trend)
tab_over, tab_prod, tab_seg, tab_ai, tab_data = st.tabs(
    ["Overview", "Products", "Regions & categories", "AI insights", "Data"]
)

with tab_over:
    with panel("Revenue trend", "Monthly revenue. Orange markers flag months whose growth is a statistical outlier."):
        show(ch.revenue_trend(trend, anomalies))
    left, right = st.columns(2)
    with left:
        with panel("Monthly profit", "Profit by month."):
            show(ch.profit_bars(trend))
    with right:
        with panel("Margin by discount level", "How much of each revenue dollar is kept as profit, by discount depth."):
            show(ch.margin_by_discount(m.discount_impact(df)))

with tab_prod:
    c1, c2 = st.columns([1, 2])
    n = c1.slider("Products to show", 5, 20, 10)
    by_label = c2.radio("Rank by", ["Revenue", "Profit"], horizontal=True)
    by = "Sales" if by_label == "Revenue" else "Profit"
    with panel(f"Top {n} products by {by_label.lower()}", "Hover a bar for exact values and margin."):
        show(ch.top_products(m.top_products(df, n, by), by))
    with panel("Profit by sub-category", "Bars to the left of zero lose money."):
        show(ch.subcategory_profit(m.breakdown(df, "Sub-Category")))

with tab_seg:
    # colours are tied to the entity over the FULL dataset, so filtering never repaints a category
    cmap = dict(zip(all_cats, ch.SERIES)) if len(all_cats) <= len(ch.SERIES) else None
    left, right = st.columns(2)
    with left:
        with panel("Revenue by region", "Hover for profit and margin."):
            show(ch.entity_bar(m.breakdown(df, "Region"), "Region"))
    with right:
        with panel("Revenue by category", "Each category keeps its colour when you filter."):
            show(ch.entity_bar(m.breakdown(df, "Category"), "Category", cmap))
    st.markdown("**Revenue by region and category**")
    pivot = df.pivot_table(index="Region", columns="Category", values="Sales", aggfunc="sum", fill_value=0)
    st.dataframe(pivot.style.format("${:,.0f}"), **_stretch(st.dataframe))

with tab_ai:
    summary = m.build_summary(df)
    st.markdown(
        "The numbers below are computed with pandas. The writer (AI or rule-based) only **explains** them; "
        "it never does the calculations."
    )
    state_key = (start, end, tuple(regions), tuple(categories))
    if st.button("Generate insights", type="primary"):
        with st.spinner("Writing insights..."):
            text, source = generate_insights(summary)
        st.session_state["insights"] = {"key": state_key, "text": text, "source": source}

    saved = st.session_state.get("insights")
    if saved and saved["key"] == state_key:
        with st.container(border=True):
            st.markdown(saved["text"])
        st.caption(f"Writer: {saved['source']}")
        bad = ungrounded_numbers(saved["text"], summary)
        if bad:
            st.warning("These figures were not found in the computed metrics, so verify them: " + ", ".join(bad))
        else:
            st.caption("Check passed: every figure quoted was found in the computed metrics.")
    elif saved:
        st.info("Filters changed since the last run. Click **Generate insights** to refresh.")

    with st.expander("What the writer sees (computed metrics)"):
        st.json(summary)

with tab_data:
    shown = df.copy()
    shown["Order Date"] = shown["Order Date"].dt.date
    st.dataframe(shown, hide_index=True, **_stretch(st.dataframe))
    st.download_button("Download filtered data (CSV)", shown.to_csv(index=False), "filtered_sales.csv", "text/csv")

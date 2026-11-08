"""Pure analysis functions: no Streamlit, no LLM. Everything here is unit-tested.

Design rule: ALL numbers in this project are computed here with pandas.
The LLM only receives these numbers and explains them; it never calculates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DISCOUNT_BINS = [-0.001, 0.0, 0.10, 0.20, 0.30, 0.50, 1.0]
DISCOUNT_LABELS = ["0%", "1-10%", "11-20%", "21-30%", "31-50%", ">50%"]
TREND_COLUMNS = ["Order Date", "Sales", "Profit", "Orders", "MoM Growth %"]


# --------------------------------------------------------------------------- KPIs
def compute_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"revenue": 0.0, "profit": 0.0, "orders": 0, "aov": 0.0, "margin_pct": 0.0, "units": 0}
    revenue = float(df["Sales"].sum())
    profit = float(df["Profit"].sum())
    orders = int(df["Order ID"].nunique())
    return {
        "revenue": revenue,
        "profit": profit,
        "orders": orders,
        "aov": revenue / orders if orders else 0.0,
        "margin_pct": profit / revenue * 100 if revenue else 0.0,
        "units": int(df["Quantity"].sum()),
    }


def pct_change(current: float, previous: float | None) -> float | None:
    """Percent change, or None when there is no meaningful previous value."""
    if previous is None or pd.isna(previous) or previous == 0:
        return None
    return float((current - previous) / abs(previous) * 100)


def previous_period(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows from the window of equal length that ends right before `start`."""
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    span = (end - start) + pd.Timedelta(days=1)
    return df[(df["Order Date"] >= start - span) & (df["Order Date"] < start)]


# ------------------------------------------------------------------------ Trends
def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TREND_COLUMNS)
    trend = (
        df.set_index("Order Date")
        .resample("MS")
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique"))
        .reset_index()
    )
    trend["MoM Growth %"] = (trend["Sales"].pct_change() * 100).replace([np.inf, -np.inf], np.nan)
    return trend


def detect_anomalies(trend: pd.DataFrame, z_threshold: float = 2.0, min_points: int = 8) -> pd.DataFrame:
    """Flag months whose month-over-month growth is a statistical outlier (|z| >= threshold).

    This is a simple screening tool, not proof of a problem: seasonal peaks can be flagged too.
    """
    empty = pd.DataFrame(columns=["Order Date", "Sales", "MoM Growth %", "z_score"])
    if trend.empty:
        return empty
    growth = trend["MoM Growth %"].dropna()
    if len(growth) < min_points:
        return empty
    std = growth.std(ddof=0)
    if std == 0:
        return empty
    z = (growth - growth.mean()) / std
    hit = z[z.abs() >= z_threshold]
    out = trend.loc[hit.index, ["Order Date", "Sales", "MoM Growth %"]].copy()
    out["z_score"] = hit.round(2)
    return out.reset_index(drop=True)


# -------------------------------------------------------------------- Breakdowns
def breakdown(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Revenue, profit, orders and margin for each value of `col`, biggest revenue first."""
    g = df.groupby(col, as_index=False).agg(
        Sales=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "nunique")
    )
    g["Margin %"] = np.where(g["Sales"] != 0, g["Profit"] / g["Sales"] * 100, 0.0)
    return g.sort_values("Sales", ascending=False).reset_index(drop=True)


def top_products(df: pd.DataFrame, n: int = 10, by: str = "Sales") -> pd.DataFrame:
    return breakdown(df, "Product Name").sort_values(by, ascending=False).head(n).reset_index(drop=True)


def discount_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Profit margin grouped by how deep the discount on the order line was."""
    bucket = pd.cut(df["Discount"], bins=DISCOUNT_BINS, labels=DISCOUNT_LABELS)
    g = (
        df.assign(Bucket=bucket)
        .groupby("Bucket", observed=True)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Lines=("Order ID", "size"))
        .reset_index()
    )
    g["Margin %"] = np.where(g["Sales"] != 0, g["Profit"] / g["Sales"] * 100, 0.0)
    g["Bucket"] = g["Bucket"].astype(str)
    return g


# ----------------------------------------------------------- Summary for the LLM
def _records(bd: pd.DataFrame, n: int | None = None) -> list[dict]:
    first_col = bd.columns[0]
    rows = bd if n is None else bd.head(n)
    return [
        {
            "name": str(r[first_col]),
            "revenue": round(float(r["Sales"]), 2),
            "profit": round(float(r["Profit"]), 2),
            "margin_pct": round(float(r["Margin %"]), 2),
        }
        for _, r in rows.iterrows()
    ]


def build_summary(df: pd.DataFrame) -> dict:
    """Everything the insight writer is allowed to know, as plain JSON-friendly data."""
    kpis = {k: round(v, 2) if isinstance(v, float) else v for k, v in compute_kpis(df).items()}
    if df.empty:
        return {"period": None, "kpis": kpis}

    trend = monthly_trend(df)
    best = trend.loc[trend["Sales"].idxmax()]
    worst = trend.loc[trend["Sales"].idxmin()]
    anomalies = detect_anomalies(trend)
    subcats = breakdown(df, "Sub-Category")
    losers = subcats[subcats["Profit"] < 0].sort_values("Profit").head(5)

    return {
        "period": {
            "start": df["Order Date"].min().strftime("%Y-%m-%d"),
            "end": df["Order Date"].max().strftime("%Y-%m-%d"),
            "order_lines": int(len(df)),
        },
        "kpis": kpis,
        "best_month": {"month": best["Order Date"].strftime("%Y-%m"), "revenue": round(float(best["Sales"]), 2)},
        "worst_month": {"month": worst["Order Date"].strftime("%Y-%m"), "revenue": round(float(worst["Sales"]), 2)},
        "unusual_months": [
            {
                "month": r["Order Date"].strftime("%Y-%m"),
                "revenue": round(float(r["Sales"]), 2),
                "mom_growth_pct": round(float(r["MoM Growth %"]), 2),
            }
            for _, r in anomalies.iterrows()
        ],
        "top_products": _records(top_products(df, 5)),
        "by_region": _records(breakdown(df, "Region")),
        "by_category": _records(breakdown(df, "Category")),
        "loss_making_subcategories": [
            {"name": str(r["Sub-Category"]), "revenue": round(float(r["Sales"]), 2), "profit": round(float(r["Profit"]), 2)}
            for _, r in losers.iterrows()
        ],
        "discount_buckets": [
            {
                "bucket": r["Bucket"],
                "revenue": round(float(r["Sales"]), 2),
                "profit": round(float(r["Profit"]), 2),
                "margin_pct": round(float(r["Margin %"]), 2),
            }
            for _, r in discount_impact(df).iterrows()
        ],
    }

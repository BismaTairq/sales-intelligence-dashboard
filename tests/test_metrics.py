import json

import pandas as pd
import pytest

from src import metrics as m

COLUMNS = ["Order ID", "Order Date", "Region", "Category", "Sub-Category", "Product Name",
           "Sales", "Quantity", "Discount", "Profit"]


@pytest.fixture
def df():
    """Tiny dataset where every answer can be worked out by hand."""
    rows = [
        ("O1", "2024-01-10", "West", "Furniture", "Chairs", "Chair A", 100, 1, 0.0, 20),
        ("O1", "2024-01-10", "West", "Furniture", "Chairs", "Chair B", 50, 1, 0.0, 10),   # same order, 2nd line
        ("O2", "2024-01-20", "East", "Technology", "Phones", "Phone A", 200, 2, 0.3, -20),
        ("O3", "2024-02-05", "West", "Furniture", "Tables", "Table A", 300, 1, 0.0, 60),
    ]
    d = pd.DataFrame(rows, columns=COLUMNS)
    d["Order Date"] = pd.to_datetime(d["Order Date"])
    return d


def test_kpis(df):
    k = m.compute_kpis(df)
    assert k["revenue"] == 650
    assert k["profit"] == 70
    assert k["orders"] == 3          # O1 has two lines but is ONE order
    assert k["aov"] == pytest.approx(650 / 3)
    assert k["margin_pct"] == pytest.approx(70 / 650 * 100)
    assert k["units"] == 5


def test_kpis_empty_does_not_crash(df):
    k = m.compute_kpis(df.iloc[0:0])
    assert k["revenue"] == 0 and k["orders"] == 0 and k["margin_pct"] == 0


def test_monthly_trend(df):
    t = m.monthly_trend(df)
    assert t["Sales"].tolist() == [350, 300]
    assert t["Orders"].tolist() == [2, 1]
    assert pd.isna(t["MoM Growth %"].iloc[0])
    assert t["MoM Growth %"].iloc[1] == pytest.approx(-50 / 350 * 100)


def test_monthly_trend_empty(df):
    assert m.monthly_trend(df.iloc[0:0]).empty


def test_breakdown_and_top_products(df):
    region = m.breakdown(df, "Region")
    assert region.iloc[0]["Region"] == "West" and region.iloc[0]["Sales"] == 450
    top = m.top_products(df, n=2)
    assert top["Product Name"].tolist() == ["Table A", "Phone A"]
    by_profit = m.top_products(df, n=1, by="Profit")
    assert by_profit.iloc[0]["Product Name"] == "Table A"


def test_discount_impact(df):
    d = m.discount_impact(df).set_index("Bucket")
    assert d.loc["0%", "Sales"] == 450
    assert d.loc["21-30%", "Margin %"] == pytest.approx(-10.0)


def test_pct_change_handles_zero_and_none():
    assert m.pct_change(110, 100) == pytest.approx(10.0)
    assert m.pct_change(5, 0) is None
    assert m.pct_change(5, None) is None


def test_previous_period(df):
    prev = m.previous_period(df, "2024-02-01", "2024-02-28")   # 28-day window: Jan 4 - Jan 31
    assert prev["Sales"].sum() == 350


def test_detect_anomalies_needs_enough_data(df):
    assert m.detect_anomalies(m.monthly_trend(df)).empty


def test_detect_anomalies_flags_a_spike():
    months = pd.date_range("2023-01-01", periods=14, freq="MS")
    sales = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 300]
    trend = pd.DataFrame({"Order Date": months, "Sales": sales})
    trend["MoM Growth %"] = trend["Sales"].pct_change() * 100
    out = m.detect_anomalies(trend)
    assert out["Order Date"].iloc[-1] == months[-1]


def test_build_summary_is_json_serialisable_and_consistent(df):
    s = m.build_summary(df)
    json.dumps(s)  # must not raise (no Timestamps / numpy types left)
    assert s["kpis"]["revenue"] == 650
    assert s["period"]["start"] == "2024-01-10"
    assert s["best_month"]["month"] == "2024-01"
    assert s["loss_making_subcategories"][0]["name"] == "Phones"


def test_build_summary_empty(df):
    s = m.build_summary(df.iloc[0:0])
    assert s["period"] is None

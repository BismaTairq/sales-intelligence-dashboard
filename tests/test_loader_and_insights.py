import io

import pandas as pd
import pytest

from src.data_loader import clean_data, load_data
from src.insights import rule_based_insights, ungrounded_numbers
from src.metrics import build_summary

CSV = """order_id,order date,Region,category,sub-category,Product Name,Sales,Quantity,Discount,Profit
A1,01/05/2024,West,Furniture,Chairs, Chair A ,100,1,0,20
A1,01/05/2024,West,Furniture,Chairs, Chair A ,100,1,0,20
A2,not a date,East,Technology,Phones,Phone A,50,1,0,5
A3,02/10/2024,East,Technology,Phones,Phone A,200,2,0.3,-20
"""


def test_loader_renames_columns_drops_duplicates_and_bad_dates():
    df = load_data(io.StringIO(CSV))
    assert len(df) == 2                                   # duplicate and unparseable date removed
    assert "Order Date" in df.columns and "Sub-Category" in df.columns
    assert df["Order Date"].dtype.kind == "M"             # real datetimes
    assert df["Product Name"].iloc[0] == "Chair A"        # whitespace trimmed


def test_loader_reports_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_data(pd.DataFrame({"Order ID": [1], "Sales": [10]}))


def _summary():
    df = load_data(io.StringIO(CSV))
    return build_summary(df)


def test_grounding_accepts_real_numbers_and_flags_invented_ones():
    s = _summary()
    good = "Revenue was $300 with a profit of $0 across 2 orders."
    assert ungrounded_numbers(good, s) == []
    bad = "Revenue grew 87.5% and reached $999,999."
    flagged = ungrounded_numbers(bad, s)
    assert "87.5%" in flagged
    assert any("999,999" in f for f in flagged)


def test_rule_based_insights_only_quotes_known_numbers():
    s = _summary()
    text = rule_based_insights(s)
    assert "Key findings" in text
    assert ungrounded_numbers(text, s) == []

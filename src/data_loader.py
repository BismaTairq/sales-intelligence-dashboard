"""Load and clean a Superstore-style sales CSV.

Cleaning rules (kept simple and explicit so you can explain each one):
1. Normalise column names ("order_date", "Order date" -> "Order Date").
2. Fail early, with a clear message, if a required column is missing.
3. Parse dates (month-first, like the Kaggle Superstore file) and numbers;
   unparseable values become NaN.
4. Drop rows missing a date, order id, sales or profit, and exact duplicates.
5. Trim whitespace in text columns and sort by date.
"""
from __future__ import annotations

import re

import pandas as pd

REQUIRED_COLUMNS = [
    "Order ID",
    "Order Date",
    "Region",
    "Category",
    "Sub-Category",
    "Product Name",
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
]
OPTIONAL_COLUMNS = ["Row ID", "Ship Date", "Segment", "Customer Name", "Country", "State", "City"]


def _key(name: object) -> str:
    """'Sub-Category', 'sub_category', 'SUB CATEGORY' -> 'subcategory'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_CANONICAL = {_key(c): c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_CANONICAL.get(_key(c), str(c).strip()) for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Columns found: {list(df.columns)}"
        )

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", format="mixed")
    for col in ["Sales", "Quantity", "Discount", "Profit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Order ID", "Order Date", "Sales", "Profit"])
    # A missing discount means "no discount"; a missing quantity is counted as 0 units.
    df["Discount"] = df["Discount"].fillna(0.0)
    df["Quantity"] = df["Quantity"].fillna(0)

    df = df.drop_duplicates()
    for col in ["Region", "Category", "Sub-Category", "Product Name"]:
        df[col] = df[col].astype(str).str.strip()

    return df.sort_values("Order Date").reset_index(drop=True)


def load_data(source) -> pd.DataFrame:
    """`source` can be a file path or a file-like object (e.g. an uploaded CSV)."""
    # latin-1 reads the Kaggle Superstore file, which is not valid UTF-8.
    raw = pd.read_csv(source, encoding="latin-1")
    return clean_data(raw)

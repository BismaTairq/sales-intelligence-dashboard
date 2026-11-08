"""Generate a synthetic Superstore-style sales file so the dashboard runs with no download.

The data is made up (fictional brands, random numbers) but has realistic patterns:
seasonality, yearly growth, regional differences, and discounts that hurt margin.
Replace it with the real Kaggle "Superstore" CSV when you are ready.

    python generate_sample_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# sub-category: (min unit price, max unit price, base profit margin)
CATALOG = {
    "Furniture": {
        "Bookcases": (120, 600, 0.05),
        "Chairs": (80, 700, 0.12),
        "Furnishings": (15, 120, 0.25),
        "Tables": (200, 1100, -0.02),
    },
    "Office Supplies": {
        "Binders": (5, 60, 0.30),
        "Paper": (6, 40, 0.40),
        "Storage": (15, 200, 0.12),
        "Art": (4, 30, 0.28),
        "Labels": (5, 25, 0.42),
        "Supplies": (10, 150, -0.02),
    },
    "Technology": {
        "Phones": (60, 900, 0.18),
        "Accessories": (10, 200, 0.28),
        "Copiers": (500, 3500, 0.35),
        "Machines": (150, 2500, 0.05),
    },
}
BRANDS = ["Acme", "Nova", "Apex", "Zenith", "Orion"]
REGIONS = {"West": 0.32, "East": 0.29, "Central": 0.21, "South": 0.18}
SEGMENTS = {"Consumer": 0.52, "Corporate": 0.30, "Home Office": 0.18}
DISCOUNTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
DISCOUNT_P = [0.52, 0.10, 0.16, 0.09, 0.06, 0.04, 0.03]
SEASON = [0.70, 0.65, 0.90, 0.85, 0.90, 0.95, 0.85, 0.90, 1.25, 1.10, 1.50, 1.60]
YEAR_GROWTH = {2022: 1.00, 2023: 1.12, 2024: 1.25, 2025: 1.40}


def generate(path: str | Path = "data/sample_sales.csv", seed: int = 42) -> Path:
    path = Path(path)
    rng = np.random.default_rng(seed)

    products = []  # (category, sub_category, name, unit_price, base_margin)
    for cat, subs in CATALOG.items():
        for sub, (lo, hi, margin) in subs.items():
            for i, brand in enumerate(BRANDS):
                price = round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 2)
                products.append((cat, sub, f"{brand} {sub} {chr(65 + i)}", price, margin))
    weights = np.array([1 / np.sqrt(p[3]) for p in products])
    weights = weights / weights.sum()

    region_names, region_p = list(REGIONS), list(REGIONS.values())
    segment_names, segment_p = list(SEGMENTS), list(SEGMENTS.values())

    rows, order_no, row_id = [], 100000, 0
    for year, growth in YEAR_GROWTH.items():
        for month in range(1, 13):
            n_orders = max(int(rng.normal(55 * SEASON[month - 1] * growth, 5)), 5)
            days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
            for _ in range(n_orders):
                order_no += 1
                order_date = pd.Timestamp(year=year, month=month, day=int(rng.integers(1, days_in_month + 1)))
                ship_date = order_date + pd.Timedelta(days=int(rng.integers(2, 7)))
                region = str(rng.choice(region_names, p=region_p))
                segment = str(rng.choice(segment_names, p=segment_p))
                for _ in range(int(rng.integers(1, 5))):
                    cat, sub, name, price, margin = products[int(rng.choice(len(products), p=weights))]
                    qty = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8], p=[0.30, 0.25, 0.18, 0.10, 0.07, 0.05, 0.03, 0.02]))
                    disc = float(rng.choice(DISCOUNTS, p=DISCOUNT_P))
                    sales = price * qty * (1 - disc)
                    profit = sales * (margin - 0.6 * disc + rng.normal(0, 0.03))
                    row_id += 1
                    rows.append(
                        {
                            "Row ID": row_id,
                            "Order ID": f"US-{year}-{order_no}",
                            "Order Date": order_date.strftime("%m/%d/%Y"),
                            "Ship Date": ship_date.strftime("%m/%d/%Y"),
                            "Segment": segment,
                            "Region": region,
                            "Category": cat,
                            "Sub-Category": sub,
                            "Product Name": name,
                            "Sales": round(sales, 2),
                            "Quantity": qty,
                            "Discount": disc,
                            "Profit": round(profit, 2),
                        }
                    )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = generate()
    df = pd.read_csv(out)
    print(f"Wrote {len(df):,} rows to {out}")

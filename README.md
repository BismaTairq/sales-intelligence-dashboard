# Sales Intelligence Dashboard

An interactive dashboard that analyses sales data, highlights trends, and writes plain-English business insights.
Numbers are computed with pandas; an optional LLM (Claude) only **explains** them, so it cannot invent statistics.

> Status: portfolio prototype built on synthetic data. Not production-ready (see Limitations).

## Problem and users

Sales managers and small-business owners have order data in spreadsheets but little time to dig through it.
They need to see what is selling, what is shrinking, where profit leaks, and what to look into next.

## Features (MVP)

- KPI cards: revenue, profit, orders, average order value, profit margin, each compared with the previous period
- Charts: monthly revenue trend with outlier months flagged, monthly profit, margin by discount level,
  top products, profit by sub-category, revenue by region and category
- Filters: date range, region, category (empty selections are handled gracefully)
- "Generate insights" button: AI-written (with an API key) or rule-based (without one)
- Grounding check: every number quoted in the insight text is compared with the computed metrics and unmatched
  figures are flagged
- Upload your own Superstore-style CSV, or download the filtered data

## Architecture

```
CSV -> data_loader (clean) -> metrics (pandas) --+--> charts (Plotly) --> Streamlit dashboard
                                                 |
                                                 +--> summary (JSON) --> insights (Claude or rules)
                                                                              |
                                                          grounding check <---+
```

| File | Job |
|---|---|
| `app.py` | Streamlit layout, filters, tabs |
| `src/data_loader.py` | Load CSV, normalise columns, parse dates, drop bad rows |
| `src/metrics.py` | All calculations (KPIs, trends, breakdowns, anomalies, summary) |
| `src/charts.py` | Plotly figures with one consistent style |
| `src/insights.py` | LLM writer, rule-based writer, number-grounding check |
| `generate_sample_data.py` | Creates synthetic `data/sample_sales.csv` |
| `tests/` | pytest unit tests |

## Setup (VS Code)

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python generate_sample_data.py      # optional; the app creates it if missing
streamlit run app.py
```

Optional AI insights: paste an Anthropic API key in the sidebar, or set it before launching:

```bash
# Windows PowerShell:  $env:ANTHROPIC_API_KEY="your-key"
# macOS/Linux:         export ANTHROPIC_API_KEY="your-key"
```

## Using the real Superstore dataset

Download "Superstore Sales" from Kaggle, then upload the CSV in the sidebar. Required columns:
`Order ID, Order Date, Region, Category, Sub-Category, Product Name, Sales, Quantity, Discount, Profit`.
Dates are read month-first (US format), as in the Kaggle file.

## Tests

```bash
pytest
```

Tests use a tiny hand-made dataframe with known answers, plus checks for empty data, bad dates, missing columns,
and the number-grounding check.

## Limitations

- Prototype on synthetic data; no authentication, no real-time data, single user.
- Insights need human review. The grounding check only confirms a number exists in the metrics (within rounding);
  it does not verify that the sentence around it is correct.
- The outlier flag is a simple z-score on month-over-month growth; it can flag normal seasonal peaks.
- The first and last month of a selection may be partial months, which distorts growth figures.
- The comparison with the "previous period" uses a window of equal length; it is empty when the selection starts
  at the beginning of the data.
- Forecasting and scheduled reports are not included yet.

## Next steps

1. Run it on the real Superstore data and note what surprises you.
2. Add a simple forecast (moving average first, then a time-series model) and compare against held-out months.
3. Evaluate insight quality: have a few people rate usefulness, and track how often the grounding check fires.
4. Later: scheduled weekly report (n8n), PostgreSQL instead of CSV, authentication.

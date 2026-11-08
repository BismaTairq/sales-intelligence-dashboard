"""Turn the computed metrics into business-language insights.

Two writers share one input (the summary dict from metrics.build_summary):
- an LLM writer (Claude) used when an API key is available;
- a rule-based writer used otherwise, so the app always works offline.

`ungrounded_numbers` is the safety check: it lists numbers in the text that cannot
be found in the metrics we supplied (a cheap guard against invented statistics).
"""
from __future__ import annotations

import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are a careful sales analyst writing for a busy business owner. "
    "You will receive a JSON summary of sales metrics that were already computed with code."
)

USER_PROMPT = """Here are the computed sales metrics (JSON):

{summary}

Write the insights with this structure:
### Key findings
5 short bullets about what the numbers show (trends, strongest and weakest areas, profit problems).
### Recommended actions
3 short bullets: concrete things to investigate or try next. Present causes as hypotheses, not facts.
### Caveat
One sentence on what these numbers cannot tell us.

Hard rules:
- Use ONLY numbers that appear in the JSON, quoted exactly as given (for example 12.34%, $2,297,201.50). Do not compute new numbers, round, or convert to K/M.
- Do not invent products, regions, customers or causes that are not in the JSON.
- Plain language, no jargon."""


def build_prompt(summary: dict) -> str:
    return USER_PROMPT.format(summary=json.dumps(summary, indent=2))


# ------------------------------------------------------------------ rule-based writer
def _money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def rule_based_insights(s: dict) -> str:
    if not s.get("period"):
        return "No data in the current selection."
    k = s["kpis"]
    out = ["### Key findings"]
    out.append(
        f"- **Overall:** {_money(k['revenue'])} revenue and {_money(k['profit'])} profit from "
        f"{k['orders']:,} orders, a profit margin of {k['margin_pct']:.1f}%."
    )
    out.append(
        f"- **Months:** the strongest month was {s['best_month']['month']} "
        f"({_money(s['best_month']['revenue'])}); the weakest was {s['worst_month']['month']} "
        f"({_money(s['worst_month']['revenue'])})."
    )
    if s["unusual_months"]:
        u = s["unusual_months"][0]
        out.append(
            f"- **Unusual month:** {u['month']} had month-over-month growth of {u['mom_growth_pct']:.1f}%, "
            "an outlier compared with the other months."
        )

    regions = s["by_region"]
    if regions:
        top_r = regions[0]
        low_r = min(regions, key=lambda r: r["margin_pct"])
        out.append(
            f"- **Regions:** {top_r['name']} has the highest revenue ({_money(top_r['revenue'])}); "
            f"{low_r['name']} has the lowest margin ({low_r['margin_pct']:.1f}%)."
        )
    cats = s["by_category"]
    if cats:
        hi_c = max(cats, key=lambda c: c["margin_pct"])
        lo_c = min(cats, key=lambda c: c["margin_pct"])
        out.append(
            f"- **Categories:** {hi_c['name']} has the best margin ({hi_c['margin_pct']:.1f}%) and "
            f"{lo_c['name']} the weakest ({lo_c['margin_pct']:.1f}%)."
        )
    if s["loss_making_subcategories"]:
        names = ", ".join(x["name"] for x in s["loss_making_subcategories"][:3])
        worst = s["loss_making_subcategories"][0]
        out.append(f"- **Losses:** {names} lose money; {worst['name']} is worst at {_money(worst['profit'])} profit.")
    buckets = {b["bucket"]: b for b in s["discount_buckets"]}
    if "0%" in buckets and len(buckets) > 1:
        deepest = [b for b in s["discount_buckets"] if b["bucket"] != "0%"][-1]
        out.append(
            f"- **Discounts:** undiscounted lines earn a {buckets['0%']['margin_pct']:.1f}% margin, "
            f"while the {deepest['bucket']} discount group earns {deepest['margin_pct']:.1f}%."
        )

    out.append("### Things to investigate")
    if s["loss_making_subcategories"]:
        out.append("- Review pricing, costs and discounting for the loss-making sub-categories listed above.")
    if "0%" in buckets and len(buckets) > 1 and deepest["margin_pct"] < buckets["0%"]["margin_pct"]:
        out.append("- Test whether tighter approval rules for deep discounts protect margin without hurting volume.")
    if regions:
        out.append(f"- Look into why {low_r['name']} converts revenue to profit less efficiently than other regions.")
    out.append("### Caveat")
    out.append(
        "These are descriptive statistics from the selected data only; they show what happened, "
        "not why, and the suggestions above are hypotheses to test."
    )
    return "\n".join(out)


# ---------------------------------------------------------------------- LLM writer
def generate_insights(summary: dict, api_key: str | None = None) -> tuple[str, str]:
    """Return (markdown_text, source_label)."""

    key = api_key or os.getenv("GEMINI_API_KEY")

    if not key:
        return rule_based_insights(summary), "Rule-based (no API key set)"

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)

        prompt = f"""
        {SYSTEM_PROMPT}

        {build_prompt(summary)}
        """

        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL"),
            # model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=900,
                temperature=0.7,
            ),
        )

        return response.text, "AI-written (Gemini)"

    except Exception as exc:
        return (
            rule_based_insights(summary),
            f"Rule-based (AI call failed: {type(exc).__name__}: {exc})"
        )

# ------------------------------------------------------------- grounding check
_NUM = re.compile(r"(\d[\d,]*(?:\.\d+)?)([KMB](?![A-Za-z]))?(\s?%)?")


def _collect(obj, nums: list[float], strs: list[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect(v, nums, strs)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, nums, strs)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        nums.append(abs(float(obj)))
    elif isinstance(obj, str):
        strs.append(obj)


def ungrounded_numbers(text: str, summary: dict) -> list[str]:
    """Numbers quoted in `text` that do not appear (within rounding) in `summary`.

    Years and small whole numbers (ranks, counts of bullets) are ignored.
    Names from the summary are removed first so digits inside product names do not count.
    """
    nums: list[float] = []
    strs: list[str] = []
    _collect(summary, nums, strs)
    for s in strs:  # digits inside labels, e.g. "1-10%" or "2025-12"
        for m in _NUM.finditer(s):
            nums.append(float(m.group(1).replace(",", "")))

    clean = text
    for s in sorted({s for s in strs if s}, key=len, reverse=True):
        clean = clean.replace(s, " ")

    bad: list[str] = []
    for m in _NUM.finditer(clean):
        value = float(m.group(1).replace(",", ""))
        suffix, pct = m.group(2), m.group(3)
        if suffix:
            value *= {"K": 1e3, "M": 1e6, "B": 1e9}[suffix]
        if not suffix and not pct and value == int(value) and (value <= 12 or 1990 <= value <= 2100):
            continue
        tol = 0.05 if suffix else 0.006
        if not any(abs(value - n) <= max(tol * abs(n), 0.051) for n in nums):
            bad.append(m.group(0).strip())
    return bad

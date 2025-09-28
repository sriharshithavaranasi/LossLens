# utils/generate_insight.py
import os
import pandas as pd

try:
    import openai
except Exception:
    openai = None


def _sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure key columns are numeric and clean up formatting issues."""
    for col in ["Amount", "Happiness", "Regret"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _local_summary(df: pd.DataFrame) -> str:
    """Fallback if GPT not available: simple summary of regrets by category + merchants."""
    if df.empty:
        return "No spending data available."

    df = _sanitize_numeric(df)

    # Get top 5 regretful purchases
    top_merchants = df.sort_values("Regret", ascending=False).head(5)

    summary = "Here are your top regretful purchases:\n\n"
    for _, r in top_merchants.iterrows():
        date_str = (
            r["Date"].strftime("%Y-%m-%d") if "Date" in r and not pd.isna(r["Date"]) else "Unknown date"
        )
        happiness = int(r["Happiness"]) if not pd.isna(r["Happiness"]) else "N/A"
        summary += (
            f"- {r['Merchant']} on {date_str}: "
            f"${r['Amount']:,.2f}, "
            f"Happiness {happiness}, "
            f"Regret ${r['Regret']:,.2f}\n"
        )

    return summary


def _local_advice(df: pd.DataFrame) -> str:
    """Fallback advice without GPT."""
    return (
        "Here are a few ways to lower spending regret:\n"
        "- Focus on reducing large purchases in your top regret category.\n"
        "- Set a budget limit for impulse buys.\n"
        "- Reflect on happiness before spending: if it’s below 3/5, reconsider the purchase."
    )


def generate_insight(
    df: pd.DataFrame,
    use_gpt: bool = False,
    model: str = "gpt-4o",
    mode: str = "summary",
) -> str:
    """
    Generate either a summary or advice about spending.
    mode = "summary" -> short recap of habits
    mode = "advice"  -> simple, actionable budgeting tips
    """

    if df is None or df.empty:
        return "No spending data available."

    df = _sanitize_numeric(df)

    if use_gpt and openai is not None and os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Prep structured input
        by_cat = df.groupby("Category")["Regret"].sum().sort_values(ascending=False).head(10)
        top_items = df.sort_values("Regret", ascending=False).head(10)

        base_info = "Category regret totals:\n"
        for c, amt in by_cat.items():
            base_info += f"- {c}: ${amt:,.2f}\n"

        base_info += "\nTop purchases (merchant | amount | happiness | regret):\n"
        for _, row in top_items.iterrows():
            happiness = int(row["Happiness"]) if not pd.isna(row["Happiness"]) else "N/A"
            base_info += f"- {row['Merchant']} | ${row['Amount']:,.2f} | happiness {happiness} | regret ${row['Regret']:,.2f}\n"

        if mode == "summary":
            prompt = (
                "You are a friendly financial reflection assistant. "
                "Write a clear, 3–4 sentence summary of the person’s spending habits and regrets. "
                "Use plain, simple English so that anyone can understand.\n\n"
                f"{base_info}"
            )
        elif mode == "advice":
            prompt = (
                "You are a helpful budgeting coach. Based on the data below, "
                "give 2–3 actionable, practical tips to reduce regretful spending. "
                "Avoid technical jargon and keep it simple.\n\n"
                f"{base_info}"
            )
        else:
            return "Invalid mode requested."

        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=400,
            )
            text = resp.choices[0].message["content"].strip()
            return text
        except Exception as e:
            print("OpenAI insight generation failed:", e)
            return _local_summary(df) if mode == "summary" else _local_advice(df)

    # Fallback (no GPT)
    if mode == "summary":
        return _local_summary(df)
    elif mode == "advice":
        return _local_advice(df)
    else:
        return "Invalid mode requested."

# utils/categorize_gpt.py
import os
import json
import time
import re
import pandas as pd

# Try to import OpenAI; caller decides whether to use GPT
try:
    import openai
except Exception:
    openai = None

# Simple rule-based fallback mapping
FALLBACK_KEYWORDS = {
    "Groceries": ["whole foods", "grocery", "safeway", "walmart", "aldi", "kroger"],
    "Dining": ["starbucks", "restaurant", "cafe", "diner", "pizza", "burger", "taco"],
    "Transport": ["uber", "lyft", "metra", "train", "bus", "shell", "gas", "exxon"],
    "Shopping": ["amazon", "target", "best buy", "shop"],
    "Entertainment": ["netflix", "hulu", "movie", "cinema", "spotify"],
    "Health": ["pharmacy", "walgreens", "cvs", "doctor", "clinic"],
    "Travel": ["airlines", "delta", "american airlines", "hotel", "hilton"],
    "Utilities": ["electric", "water", "comcast", "verizon", "att"]
}

def heuristic_category(merchant: str) -> str:
    m = merchant.lower()
    for cat, keys in FALLBACK_KEYWORDS.items():
        for k in keys:
            if k in m:
                return cat
    return "Other"

def categorize_transactions(df: pd.DataFrame, use_gpt: bool = False, model: str = "gpt-4o") -> pd.DataFrame:
    """
    Attempts to categorize each transaction using OpenAI if use_gpt True and API key found.
    Otherwise falls back to heuristic_category.
    Returns a DataFrame with Category column filled.
    """
    df = df.copy()
    if "Category" not in df.columns:
        df["Category"] = ""

    merchants = df["Merchant"].astype(str).fillna("").unique().tolist()

    if use_gpt and openai is not None and os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # Build a single prompt asking for JSON mapping merchant -> category
        prompt = (
            "You are a helpful assistant that maps merchant names to one of these categories: "
            "Groceries, Dining, Transport, Utilities, Shopping, Health, Entertainment, Travel, Other.\n\n"
            "Respond ONLY with a JSON array like:\n"
            '[{"merchant": "Starbucks", "category": "Dining"}, ...]\n\n'
            "Here are merchant names:\n"
        )
        for m in merchants:
            prompt += f"- {m}\n"
        prompt += "\nRemember to output valid JSON only."

        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=800,
            )
            text = resp.choices[0].message["content"]
            # Attempt to parse JSON from the response
            j = json.loads(re.search(r"(\[.*\])", text, flags=re.S).group(1))
            mapping = {item["merchant"].strip(): item["category"].strip() for item in j}
            # apply
            df["Category"] = df["Merchant"].apply(lambda m: mapping.get(str(m), heuristic_category(str(m))))
            return df
        except Exception as e:
            # Any failure -> fallback to heuristic
            print("OpenAI categorization failed, using fallback. Error:", e)

    # fallback route (or if user disabled GPT)
    df["Category"] = df["Merchant"].fillna("").apply(heuristic_category)
    return df

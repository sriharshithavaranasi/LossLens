# utils/peer_compare.py
import pandas as pd
import numpy as np

# Example synthetic peer profiles (can expand later)
PEER_PROFILES = {
    "College Student": {
        "Food": 150,
        "Takeout": 120,
        "Clothes": 75,
        "Entertainment": 100,
    },
    "Young Professional": {
        "Food": 300,
        "Takeout": 200,
        "Clothes": 200,
        "Entertainment": 250,
    },
    "Family Household": {
        "Food": 600,
        "Takeout": 300,
        "Clothes": 400,
        "Entertainment": 350,
    },
}

def compare_to_peer_profile(df, profile_name="College Student"):
    """
    Compare user spending/regret to a peer profile.
    Returns (comparison_df, sentences)
    """
    if df is None or df.empty:
        return pd.DataFrame(), ["No data available for comparison."]

    if profile_name not in PEER_PROFILES:
        return pd.DataFrame(), [f"Profile '{profile_name}' not found."]

    profile = PEER_PROFILES[profile_name]

    # Aggregate user monthly spend per category
    df["month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby(["month","Category"]).agg(
        spend=("Amount","sum"),
        regret=("Regret","sum")
    ).reset_index()

    user_cat = monthly.groupby("Category").agg(
        user_avg_monthly_spend=("spend","mean"),
        user_regret_ratio=("regret","sum")
    ).reset_index()

    # normalize regret ratio
    user_cat["user_regret_ratio"] = user_cat["user_regret_ratio"] / user_cat["user_avg_monthly_spend"].replace(0,np.nan)

    rows = []
    sentences = []

    for cat, peer_avg in profile.items():
        user_row = user_cat[user_cat["Category"] == cat]
        if not user_row.empty:
            u_spend = float(user_row["user_avg_monthly_spend"].values[0])
            u_regret_ratio = float(user_row["user_regret_ratio"].values[0])
        else:
            u_spend = 0.0
            u_regret_ratio = 0.0

        pct_diff_spend = ((u_spend - peer_avg) / peer_avg * 100) if peer_avg > 0 else None
        peer_regret_ratio = 0.2  # assume peers regret ~20% of spend (synthetic baseline)
        pct_diff_regret_ratio = ((u_regret_ratio - peer_regret_ratio) / peer_regret_ratio * 100) if peer_regret_ratio > 0 else None

        rows.append({
            "Category": cat,
            "user_avg_monthly_spend": u_spend,
            "peer_avg_monthly_spend": peer_avg,
            "pct_diff_spend": pct_diff_spend,
            "user_regret_ratio": u_regret_ratio,
            "peer_regret_ratio": peer_regret_ratio,
            "pct_diff_regret_ratio": pct_diff_regret_ratio
        })

        # Insight sentence
        if pct_diff_spend is not None:
            if pct_diff_spend > 20:
                sentences.append(f"You spend about {pct_diff_spend:.0f}% more on {cat} than the average {profile_name}, and regret {pct_diff_regret_ratio:.0f}% more often.")
            elif pct_diff_spend < -20:
                sentences.append(f"You spend about {abs(pct_diff_spend):.0f}% less on {cat} than the average {profile_name}, and regret {pct_diff_regret_ratio:.0f}% less often.")
            else:
                sentences.append(f"Your {cat} spending is roughly in line with {profile_name}s.")

    comp_df = pd.DataFrame(rows)
    return comp_df, sentences

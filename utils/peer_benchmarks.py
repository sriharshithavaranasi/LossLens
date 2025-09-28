# utils/peer_benchmarks.py
import pandas as pd

# Synthetic baseline peer data: avg monthly spend and avg regret ratio per category
# NOTE: these are synthetic example numbers — replace with real benchmarks if available.
PEER_PROFILES = {
    "student": {
        "Groceries": {"avg_monthly_spend": 150, "regret_ratio": 0.12},
        "Dining": {"avg_monthly_spend": 120, "regret_ratio": 0.18},
        "Transport": {"avg_monthly_spend": 60, "regret_ratio": 0.10},
        "Shopping": {"avg_monthly_spend": 80, "regret_ratio": 0.22},
        "Entertainment": {"avg_monthly_spend": 50, "regret_ratio": 0.15},
        "Other": {"avg_monthly_spend": 40, "regret_ratio": 0.10}
    },
    "young_professional": {
        "Groceries": {"avg_monthly_spend": 300, "regret_ratio": 0.10},
        "Dining": {"avg_monthly_spend": 250, "regret_ratio": 0.16},
        "Transport": {"avg_monthly_spend": 120, "regret_ratio": 0.08},
        "Shopping": {"avg_monthly_spend": 200, "regret_ratio": 0.18},
        "Entertainment": {"avg_monthly_spend": 100, "regret_ratio": 0.12},
        "Other": {"avg_monthly_spend": 80, "regret_ratio": 0.09}
    },
    "family": {
        "Groceries": {"avg_monthly_spend": 700, "regret_ratio": 0.07},
        "Dining": {"avg_monthly_spend": 400, "regret_ratio": 0.14},
        "Transport": {"avg_monthly_spend": 250, "regret_ratio": 0.09},
        "Shopping": {"avg_monthly_spend": 350, "regret_ratio": 0.15},
        "Entertainment": {"avg_monthly_spend": 200, "regret_ratio": 0.11},
        "Other": {"avg_monthly_spend": 150, "regret_ratio": 0.09}
    }
}

def _user_category_stats(df: pd.DataFrame):
    """Return user's monthly spend and regret ratio per category (monthly aggregation)."""
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["month"] = df["Date"].dt.to_period("M")
        monthly = df.groupby(["month","Category"]).agg(month_spend=("Amount","sum"), month_regret=("Regret","sum")).reset_index()
        # aggregate across months -> average monthly spend & regret ratio per category
        cat = monthly.groupby("Category").agg(
            avg_monthly_spend=("month_spend","mean"),
            avg_monthly_regret=("month_regret","mean")
        ).reset_index()
        cat["user_regret_ratio"] = cat["avg_monthly_regret"] / cat["avg_monthly_spend"].replace(0, pd.NA)
        return cat
    else:
        # fallback: compute total over period assumed to be 1 month
        cat = df.groupby("Category").agg(
            avg_monthly_spend=("Amount","sum"),
            avg_monthly_regret=("Regret","sum")
        ).reset_index()
        cat["user_regret_ratio"] = cat["avg_monthly_regret"] / cat["avg_monthly_spend"].replace(0, pd.NA)
        return cat

def compare_to_peer_profile(df: pd.DataFrame, profile_name: str = "student"):
    """
    Compare user's category spend/regret with peer profile.
    Returns:
      - comparison_df: DataFrame with category, user_spend, peer_spend, pct_diff_spend, user_regret_ratio, peer_regret_ratio, pct_diff_regret
      - summary_sentences: list of short human readable sentences
    """
    if profile_name not in PEER_PROFILES:
        raise ValueError("Unknown profile")

    profile = PEER_PROFILES[profile_name]
    user_stats = _user_category_stats(df)
    # build results
    rows = []
    for cat in set(list(user_stats["Category"]) + list(profile.keys())):
        user_row = user_stats[user_stats["Category"]==cat]
        user_spend = float(user_row["avg_monthly_spend"].iloc[0]) if not user_row.empty else 0.0
        user_ratio = float(user_row["user_regret_ratio"].iloc[0]) if (not user_row.empty and pd.notna(user_row["user_regret_ratio"].iloc[0])) else None
        peer_spend = profile.get(cat, {}).get("avg_monthly_spend", 0.0)
        peer_ratio = profile.get(cat, {}).get("regret_ratio", None)

        pct_diff_spend = None
        if peer_spend and peer_spend != 0:
            pct_diff_spend = (user_spend - peer_spend) / peer_spend * 100.0

        pct_diff_regret = None
        if peer_ratio and user_ratio is not None:
            pct_diff_regret = (user_ratio - peer_ratio) / peer_ratio * 100.0 if peer_ratio != 0 else None

        rows.append({
            "Category": cat,
            "user_avg_monthly_spend": user_spend,
            "peer_avg_monthly_spend": peer_spend,
            "pct_diff_spend": pct_diff_spend,
            "user_regret_ratio": user_ratio,
            "peer_regret_ratio": peer_ratio,
            "pct_diff_regret_ratio": pct_diff_regret
        })

    comp_df = pd.DataFrame(rows)
    comp_df = comp_df.sort_values(by="user_avg_monthly_spend", ascending=False)
    # Build human-readable sentences (short)
    sentences = []
    for _, r in comp_df.iterrows():
        cat = r["Category"]
        if r["peer_avg_monthly_spend"] and r["user_avg_monthly_spend"]:
            pct = r["pct_diff_spend"]
            if pct is not None:
                if pct > 20:
                    sentences.append(f"You spend {pct:.0f}% more than a typical {profile_name.replace('_',' ')} on {cat}.")
                elif pct < -20:
                    sentences.append(f"You spend {abs(pct):.0f}% less than a typical {profile_name.replace('_',' ')} on {cat}.")
        # regret ratio
        if r["peer_regret_ratio"] is not None and r["user_regret_ratio"] is not None:
            pct_r = r["pct_diff_regret_ratio"]
            if pct_r is not None:
                if pct_r > 30:
                    sentences.append(f"You regret {cat} purchases about {pct_r:.0f}% more than peers (higher regret rate).")
                elif pct_r < -30:
                    sentences.append(f"You regret {cat} purchases about {abs(pct_r):.0f}% less than peers.")
    if not sentences:
        sentences = [f"Your spending looks similar to a typical {profile_name.replace('_',' ')} in the categories tracked."]

    return comp_df, sentences

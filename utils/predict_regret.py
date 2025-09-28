# utils/predict_regret.py
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

def train_regret_model(df, min_samples=30):
    """
    Trains a simple linear regression model to predict regret from features.
    Returns (model, cv_mae) or (None, None) if not enough data.
    """
    if df is None or len(df) < min_samples:
        return None, None

    try:
        X = pd.DataFrame({
            "amount": df["Amount"].fillna(0).values,
            "happiness": df["Happiness"].fillna(3).values
        })
        y = df["Regret"].fillna(0).values

        model = LinearRegression()
        cv_mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error").mean()
        model.fit(X, y)

        return model, cv_mae
    except Exception as e:
        print("train_regret_model failed:", e)
        return None, None


def predict_regret_with_fallback(model, df, merchant, category, amount, happiness):
    """
    Predict regret for a hypothetical purchase.
    If model is provided, use it. Otherwise, fall back to heuristic.
    """
    try:
        if model is not None:
            X_new = pd.DataFrame([[amount, happiness]], columns=["amount", "happiness"])
            pred = model.predict(X_new)[0]
            return max(pred, 0.0), "model"
        else:
            # Heuristic: regret = amount × (1 - happiness/5)
            pred = amount * (1 - (happiness / 5.0))
            return round(pred, 2), "heuristic"
    except Exception as e:
        print("predict_regret_with_fallback failed:", e)
        # Final fallback
        return amount * 0.5, "default"


def compute_hotspots(df, model=None, top_n=6):
    """
    Identifies merchants and categories with highest predicted regret.
    Returns (merchant_hotspots, category_hotspots) as DataFrames.
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    try:
        # Merchant hotspots
        merchant_stats = df.groupby("Merchant").agg(
            median_amount=("Amount", "median"),
            median_happiness=("Happiness", "median"),
            count=("Merchant", "size")
        ).reset_index()

        if model is not None:
            X = merchant_stats[["median_amount", "median_happiness"]].fillna(0)
            preds = model.predict(X)
            merchant_stats["predicted_regret"] = preds
        else:
            merchant_stats["predicted_regret"] = merchant_stats["median_amount"] * (1 - (merchant_stats["median_happiness"]/5.0))

        merchant_hotspots = merchant_stats.sort_values("predicted_regret", ascending=False).head(top_n)

        # Category hotspots
        cat_stats = df.groupby("Category").agg(
            median_amount=("Amount", "median"),
            median_happiness=("Happiness", "median"),
            count=("Category", "size")
        ).reset_index()

        if model is not None:
            Xc = cat_stats[["median_amount", "median_happiness"]].fillna(0)
            preds = model.predict(Xc)
            cat_stats["predicted_regret"] = preds
        else:
            cat_stats["predicted_regret"] = cat_stats["median_amount"] * (1 - (cat_stats["median_happiness"]/5.0))

        category_hotspots = cat_stats.sort_values("predicted_regret", ascending=False).head(top_n)

        return merchant_hotspots, category_hotspots
    except Exception as e:
        print("compute_hotspots failed:", e)
        return pd.DataFrame(), pd.DataFrame()

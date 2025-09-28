# utils/regret_predictor.py
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure core columns exist and are numeric/clean."""
    df = df.copy()
    if "Amount" in df.columns:
        # keep absolute value for modeling
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").abs()
    else:
        df["Amount"] = np.nan
    df["Happiness"] = pd.to_numeric(df.get("Happiness", 3), errors="coerce").fillna(3).astype(int)
    # Ensure Regret exists
    if "Regret" not in df.columns:
        df["Regret"] = df["Amount"] * (1 - (df["Happiness"] / 5.0))
    else:
        df["Regret"] = pd.to_numeric(df["Regret"], errors="coerce")
    # Fill missing merchant/category
    df["Merchant"] = df.get("Merchant", "").fillna("Unknown").astype(str)
    df["Category"] = df.get("Category", "").fillna("Unknown").astype(str)
    # Date features
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["day_of_week"] = df["Date"].dt.dayofweek.fillna(0).astype(int)
        df["month"] = df["Date"].dt.month.fillna(0).astype(int)
    else:
        df["day_of_week"] = 0
        df["month"] = 0
    return df

def train_regret_model(df: pd.DataFrame, min_samples: int = 30):
    """
    Train a model to predict Regret from Amount, Happiness, Merchant, Category, and simple time features.
    Returns (model_pipeline, cv_mae) or (None, None) if not enough data.
    """
    df = _prepare_df(df)
    if len(df) < min_samples:
        return None, None

    feature_cols = ["Amount", "Happiness", "Category", "Merchant", "day_of_week", "month"]
    X = df[feature_cols]
    y = df["Regret"].fillna(df["Amount"] * (1 - df["Happiness"] / 5.0))

    # Column transformer
    num_cols = ["Amount", "Happiness", "day_of_week", "month"]
    cat_cols = ["Category", "Merchant"]
    preproc = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
    ], remainder="drop")

    model = Pipeline([
        ("pre", preproc),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Quick cross-validated MAE
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    cv_mae = -np.mean(scores)

    # Fit on all data
    model.fit(X, y)
    return model, float(cv_mae)

def heuristic_predict(df: pd.DataFrame, merchant: str, category: str, amount: float, happiness: int):
    """
    Fallback heuristic prediction:
    - Use historical merchant-level regret ratio if available
    - Else use category-level ratio
    - Else use (1 - happiness/5) * amount
    """
    df = _prepare_df(df)
    # Add ratio column (regret / amount)
    df = df[df["Amount"] > 0].copy()
    df["ratio"] = df["Regret"] / df["Amount"]
    # merchant mean ratio
    merchant_ratio = None
    if merchant in df["Merchant"].values:
        merchant_ratio = df[df["Merchant"] == merchant]["ratio"].mean()
    category_ratio = df[df["Category"] == category]["ratio"].mean() if category in df["Category"].values else None
    overall_ratio = df["ratio"].mean() if len(df) else None

    for candidate in (merchant_ratio, category_ratio, overall_ratio):
        if pd.notna(candidate):
            return float(amount * max(0.0, candidate))

    # Last resort: simple happiness-based proxy
    proxy = max(0.0, (1 - happiness/5.0))
    return float(amount * proxy)

def predict_regret_with_fallback(model, df, merchant: str, category: str, amount: float, happiness: int):
    """
    Use trained model if available, otherwise heuristic.
    Returns predicted_regret (float) and a 'method' string.
    """
    df = _prepare_df(df)
    if model is not None:
        # Build a single-row DataFrame with expected columns used by the pipeline
        row = {
            "Amount": amount,
            "Happiness": int(happiness),
            "Category": category if category else "Unknown",
            "Merchant": merchant if merchant else "Unknown",
            "day_of_week": 0,
            "month": 0
        }
        row_df = pd.DataFrame([row])
        try:
            pred = model.predict(row_df)[0]
            return float(max(0.0, pred)), "model"
        except Exception:
            # fallback to heuristic
            pass
    # heur
    return heuristic_predict(df, merchant, category, amount, happiness), "heuristic"

def compute_hotspots(df: pd.DataFrame, model=None, top_n:int=5):
    """
    Compute top merchants (and categories) by predicted regret.
    Strategy:
      - For each merchant, take median purchase amount (or mean) and the most common category, then predict regret.
      - Return two DataFrames: merchant_hotspots, category_hotspots
    """
    df = _prepare_df(df)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Merchant-level representative amount and category
    merchant_stats = df.groupby("Merchant").agg(
        median_amount=("Amount", "median"),
        common_category=("Category", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        count=("Amount", "size")
    ).reset_index()

    merchant_stats["predicted_regret"], merchant_stats["method"] = zip(*merchant_stats.apply(
        lambda r: predict_regret_with_fallback(model, df, r["Merchant"], r["common_category"], float(r["median_amount"] or 0.0), int(df[df["Merchant"]==r["Merchant"]]["Happiness"].median() if not df[df["Merchant"]==r["Merchant"]].empty else 3))[0:2],
        axis=1
    ))

    merchant_hotspots = merchant_stats.sort_values("predicted_regret", ascending=False).head(top_n)

    # Category-level: compute median amount per category and predict
    cat_stats = df.groupby("Category").agg(
        median_amount=("Amount", "median"),
        count=("Amount","size")
    ).reset_index()
    cat_stats["predicted_regret"], cat_stats["method"] = zip(*cat_stats.apply(
        lambda r: predict_regret_with_fallback(model, df, "", r["Category"], float(r["median_amount"] or 0.0), int(df[df["Category"]==r["Category"]]["Happiness"].median() if not df[df["Category"]==r["Category"]].empty else 3))[0:2],
        axis=1
    ))
    category_hotspots = cat_stats.sort_values("predicted_regret", ascending=False).head(top_n)

    return merchant_hotspots, category_hotspots

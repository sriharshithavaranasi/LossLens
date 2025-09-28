# utils/charting.py
import plotly.express as px
import pandas as pd

def top5_regret_bar(df: pd.DataFrame):
    top = df.sort_values("Regret", ascending=False).head(5)
    fig = px.bar(top, x="Merchant", y="Regret", hover_data=["Amount", "Happiness", "Category"],
                 title="Top 5 regretful purchases")
    fig.update_layout(yaxis_title="Regret ($)")
    return fig

def pie_regret_by_category(df: pd.DataFrame):
    by_cat = df.groupby("Category")["Regret"].sum().reset_index()
    if by_cat["Regret"].sum() == 0:
        # avoid empty pie
        fig = px.pie(names=["No regret data"], values=[1], title="Regret by category")
        return fig
    fig = px.pie(by_cat, names="Category", values="Regret", title="Regret by category")
    return fig

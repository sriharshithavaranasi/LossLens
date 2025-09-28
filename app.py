# app.py
import os
import io
import base64
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# helpers
from utils.charting import top5_regret_bar, pie_regret_by_category
from utils.categorize_gpt import categorize_transactions
from utils.generate_insight import generate_insight
from utils.regret_predictor import train_regret_model, predict_regret_with_fallback, compute_hotspots
from utils.peer_benchmarks import PEER_PROFILES, compare_to_peer_profile

# Load .env (local dev only) - DO NOT commit .env to GitHub
load_dotenv()

st.set_page_config(page_title="LossLens", layout="wide")

st.title("Regretmatics — Reflect on Your Spending 💡")
st.markdown("Upload a CSV (Date, Merchant, Amount). Rate purchases, get a regret score and AI-powered insights.")

# Sidebar: controls & API key
st.sidebar.header("Settings")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
use_gpt = st.sidebar.checkbox("Enable OpenAI auto-categorization & insights", value=bool(OPENAI_KEY))
if use_gpt and not OPENAI_KEY:
    st.sidebar.warning("No OPENAI_API_KEY found in environment. Set it in .env or Streamlit secrets to use GPT features.")

# --- TABS ---
tabs = st.tabs([
    "📂 Upload & Transactions", 
    "📊 Analysis", 
    "🤖 AI Insights", 
    "📤 Export & Email"
])

# -------------------
# TAB 1: Upload
# -------------------
with tabs[0]:
    st.header("Upload & Preview")
    uploaded = st.file_uploader("Upload CSV file (Date, Merchant, Amount)", type=["csv"])
    if uploaded is None:
        if st.button("Use sample data"):
            uploaded = open("data/sample.csv", "rb")
        else:
            st.info("Upload a CSV or press 'Use sample data' to load sample.")
            st.stop()

    # Read CSV
    def load_transactions(uploaded_file):
        df = pd.read_csv(uploaded_file)
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        # Expected columns
        expected = {"date", "merchant", "amount"}
        if not expected.issubset(set(df.columns)):
            st.error("CSV must contain columns: Date, Merchant, Amount (case-insensitive).")
            st.stop()
        # Rename canonical
        df = df.rename(columns={"date": "Date", "merchant": "Merchant", "amount": "Amount"})
        # Parse types
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Amount"] = pd.to_numeric(df["Amount"].astype(str).str.replace(r"[^0-9.-]", "", regex=True), errors="coerce")
        # Add placeholders
        if "Category" not in df.columns:
            df["Category"] = ""
        if "Happiness" not in df.columns:
            df["Happiness"] = 3
        return df

    df = load_transactions(uploaded)

    st.subheader("Transaction preview")
    st.write("Tip: edit the table to set Happiness (1-5), or use the slider inputs below.")
    try:
        edited = st.data_editor(df, num_rows="fixed", use_container_width=True)
        df = edited.copy()
    except Exception:
        st.dataframe(df)
        st.warning("Your Streamlit version doesn't support st.data_editor; consider upgrading.")

    # Validate happiness + regret score
    df["Happiness"] = pd.to_numeric(df["Happiness"], errors="coerce").fillna(3).astype(int).clip(1,5)
    df["Regret"] = (df["Amount"].abs()) * (1 - (df["Happiness"] / 5.0))
    df["Regret"] = df["Regret"].round(2)

# -------------------
# TAB 2: Analysis
# -------------------
with tabs[1]:
    st.header("Expense Analysis")

    st.subheader("Categorization")
    col1, col2 = st.columns([1,3])
    with col1:
        if st.button("Auto-categorize (GPT)"):
            with st.spinner("Calling GPT..."):
                df = categorize_transactions(df, use_gpt=use_gpt)
                st.success("Auto-categorized (or fallback used).")
    with col2:
        st.write("Or edit categories manually in the table above.")

    # Ranked table
    st.subheader("Top regretful purchases")
    df_sorted = df.sort_values("Regret", ascending=False).reset_index(drop=True)
    st.dataframe(df_sorted[["Date", "Merchant", "Category", "Amount", "Happiness", "Regret"]])

    # Charts side by side
    st.subheader("Visualizations")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_bar = top5_regret_bar(df_sorted)
        st.plotly_chart(fig_bar, use_container_width=True)
    with chart_col2:
        fig_pie = pie_regret_by_category(df_sorted)
        st.plotly_chart(fig_pie, use_container_width=True)

# -------------------
# TAB 3: AI Insights
# -------------------
# -------------------
# TAB 3: AI Insights
# -------------------
# -------------------
# TAB 3: AI Insights (What-if & Peers)
# -------------------
with tabs[2]:
    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <h3 style="margin:0">AI Insights</h3>
                <div style="color:gray">Your personal finance coach, powered by AI</div>
            </div>
            <div>
                <!-- Date range placeholder (non-functional for now) -->
                <small style="color:gray">[Date Range ▼]</small>
            </div>
        </div>
        <hr/>
        """,
        unsafe_allow_html=True
    )

    # Train model (if possible) once per run
    model, cv_mae = train_regret_model(df, min_samples=30)

    if model is not None:
        st.info(f"Local regret model trained (CV MAE ≈ ${cv_mae:.2f}). Predictions will use the model.")
    else:
        st.info("Not enough historical data to train a local model; using heuristic predictions.")

    # -------------------------
    # Top row: Summary / What-If simulator & Hotspots
    # -------------------------
    left_col, right_col = st.columns([2,1])

    with left_col:
        st.subheader("📝 Quick Summary")
        # Use the existing summary generator to give a quick plain-English summary
        try:
            quick_summary = generate_insight(df, use_gpt=use_gpt, mode="summary")
        except Exception:
            quick_summary = "Summary currently unavailable."
        st.markdown(quick_summary)

        # What-If Simulator
        st.subheader("🔮 What-If Simulator — Predict regret before you buy")
        with st.form("whatif_form"):
            m_merchant = st.selectbox("Merchant", options=[""] + sorted(df["Merchant"].dropna().unique().tolist()), index=0)
            m_category = st.selectbox("Category", options=[""] + sorted(df["Category"].dropna().unique().tolist()), index=0)
            m_amount = st.number_input("Amount ($)", min_value=0.0, value=50.0, step=1.0, format="%.2f")
            m_happiness = st.slider("Expected happiness (1 = low, 5 = high)", 1, 5, 3)
            use_model_checkbox = st.checkbox("Use trained model if available", value=True)
            submitted = st.form_submit_button("Predict regret")
        if submitted:
            model_to_use = model if use_model_checkbox else None
            pred_regret, method = predict_regret_with_fallback(model_to_use, df, m_merchant, m_category, float(m_amount), int(m_happiness))
            pct = (pred_regret / m_amount * 100.0) if m_amount > 0 else 0.0
            st.metric(label="Predicted regret ($)", value=f"${pred_regret:,.2f}", delta=f"{pct:.0f}% of purchase")
            # risk meter
            if pct >= 60:
                level = "High 🔥"
            elif pct >= 25:
                level = "Medium ⚠️"
            else:
                level = "Low ✅"
            st.write(f"Regret risk: **{level}** (predicted {pct:.0f}% of the purchase may be regretted).")
            # quick actionable tip
            if pct >= 60:
                st.warning("This purchase is likely to be highly regretted. Consider waiting 24 hours or reducing the amount.")
            elif pct >= 25:
                st.info("Moderate risk — consider if you really need it or if a cheaper option exists.")
            else:
                st.success("Low risk — looks like this purchase aligns with your past happiness.")

    with right_col:
        st.subheader("🔥 Hotspots — where regret concentrates")
        merchant_hotspots, category_hotspots = compute_hotspots(df, model=model, top_n=6)
        if not merchant_hotspots.empty:
            st.markdown("**Top merchants by predicted regret**")
            # show a small table
            display = merchant_hotspots[["Merchant","median_amount","predicted_regret","count"]].copy()
            display = display.rename(columns={"median_amount":"typical amount","predicted_regret":"predicted_regret($)","count":"tx_count"})
            display["predicted_regret($)"] = display["predicted_regret($)"].map(lambda x: f"${x:,.2f}")
            st.table(display)
        else:
            st.info("Not enough data to compute hotspots.")

        if not category_hotspots.empty:
            st.markdown("**Top categories by predicted regret**")
            cat_display = category_hotspots[["Category","median_amount","predicted_regret"]].copy()
            cat_display = cat_display.rename(columns={"median_amount":"typical amount","predicted_regret":"predicted_regret($)"})
            cat_display["predicted_regret($)"] = cat_display["predicted_regret($)"].map(lambda x: f"${x:,.2f}")
            st.table(cat_display)

    # -------------------------
    # Bottom row: Peer comparison
    # -------------------------
    st.subheader("🔎 Peer comparison (anonymized)")
    col_a, col_b = st.columns([2,1])
    with col_a:
        sel_profile = st.selectbox("Compare to:", options=list(PEER_PROFILES.keys()), index=0)
        if st.button("Compare to peers"):
            comp_df, sentences = compare_to_peer_profile(df, sel_profile)
            
            # Show textual summary
            for s in sentences:
                st.write("•", s)

            # Clean up None values before formatting
            comp_df = comp_df.fillna(0)

            # Show comparison table
            st.markdown("Category comparison table:")
            st.dataframe(
                comp_df.style.format({
                    "user_avg_monthly_spend": lambda x: f"${x:,.2f}" if pd.notna(x) else "—",
                    "peer_avg_monthly_spend": lambda x: f"${x:,.2f}" if pd.notna(x) else "—",
                    "pct_diff_spend": lambda x: f"{x:.1f}%" if pd.notna(x) else "—",
                    "user_regret_ratio": lambda x: f"{x:.2%}" if pd.notna(x) else "—",
                    "peer_regret_ratio": lambda x: f"{x:.2%}" if pd.notna(x) else "—",
                    "pct_diff_regret_ratio": lambda x: f"{x:.0f}%" if pd.notna(x) else "—"
                })
            , height=250)

    with col_b:
        st.markdown("Tips")
        st.info("Peer benchmarks are synthetic defaults. Later we can replace these with real anonymized data or allow you to opt-in to crowdsourced benchmarks.")


with tabs[3]:
    st.header("Export & Share")

    # Download CSV
    csv_bytes = df_sorted.to_csv(index=False).encode("utf-8")
    st.download_button("Download summary CSV", csv_bytes, file_name="losslens_summary.csv", mime="text/csv")

    # Email section
    st.markdown("**Email summary (optional)** — provide SMTP details (for advanced users).")
    with st.expander("Email settings / send"):
        smtp_host = st.text_input("SMTP host (e.g. smtp.gmail.com)", "")
        smtp_port = st.number_input("SMTP port", value=587)
        smtp_user = st.text_input("SMTP username (email)", "")
        smtp_password = st.text_input("SMTP password / app-password", type="password")
        to_addr = st.text_input("Send to (recipient email)", "")
        if st.button("Send email"):
            if not (smtp_host and smtp_user and smtp_password and to_addr):
                st.error("Please fill SMTP host, username, password and recipient.")
            else:
                import smtplib
                from email.message import EmailMessage
                msg = EmailMessage()
                msg["Subject"] = "LossLens summary"
                msg["From"] = smtp_user
                msg["To"] = to_addr
                msg.set_content("Attached is your LossLens summary CSV.")
                msg.add_attachment(csv_bytes, maintype="text", subtype="csv", filename="losslens_summary.csv")
                try:
                    with smtplib.SMTP(smtp_host, smtp_port) as server:
                        server.starttls()
                        server.login(smtp_user, smtp_password)
                        server.send_message(msg)
                    st.success("Email sent!")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")

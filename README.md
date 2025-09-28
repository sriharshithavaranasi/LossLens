# 💸 Regretmatics — Reflect on Spending with Regret Analytics

> **A personal finance dashboard that turns your transactions into reflections.**  
> Regretmatics helps you see not only *where* your money goes, but also *how much value* those purchases brought you — and how much regret they left behind.

---

## 🌍 Problem Statement

Many people struggle with **impulsive buying**. A late-night fast food run, another online gadget, or an impulse shopping spree often feels good in the moment but leaves you thinking:  
*“Was that really worth it?”*

Traditional budgeting apps focus on *tracking spending* but rarely help users reflect on **emotional satisfaction vs. cost**. There’s a gap for tools that **measure regret, visualize patterns, and give actionable insights**.

---

## 💡 Our Solution: Regretmatics

Regretmatics introduces a simple but powerful concept: **The Regret Score.**

\[
\text{Regret Score} = \text{Amount} \times \left(1 - \frac{\text{Happiness Score}}{5}\right)
\]

- Users upload their past transactions.  
- Each purchase is given a **Happiness Score (1–5)**.  
- Regretmatics calculates regret across all purchases, uncovers patterns, and generates **AI-powered insights** to help guide better choices in the future.

✨ With Regretmatics, you don’t just see what you spent — you see **what wasn’t worth it**.

---

## 👥 Who It’s For

Regretmatics is designed for **people who want to build healthier spending habits**:  
- Those who struggle with **impulse purchases**.  
- Anyone who feels their money slips away without value.  
- Users who prefer **clear numbers and visual evidence** over vague “spend less” advice.  

It provides a **numerical, visual, and reflective** perspective that helps users make more mindful decisions.

---

## 🚀 Features

- **📂 Upload Transactions** — Import CSVs with Date, Merchant, and Amount.  
- **🤖 Auto-Categorization** — Transactions categorized by GPT (if API key available) or static fallback logic.  
- **😊 Happiness Ratings** — Rate each purchase 1–5 directly in the dashboard.  
- **📊 Regret Score Calculation** — Transparent math to quantify regret.  
- **🔎 Visual Analytics**  
  - Top 5 Regretful Purchases (bar chart)  
  - Regret by Category (pie chart)  
  - Peer comparisons and category benchmarks  
- **🧠 AI Insights** — GPT-powered natural language summaries and advice, with local fallbacks.  
- **🔮 Regret Prediction** — ML models to simulate what-if scenarios and identify regret “hotspots.”  
- **📥 Export Options** — Download results as CSV or email a summary to yourself.

---

## 🛠️ Tech Stack

**Languages & Frameworks**
- Python 3.10+  
- [Streamlit](https://streamlit.io/) — interactive web UI  

**Libraries**
- [pandas](https://pandas.pydata.org/) — data manipulation  
- [Plotly Express](https://plotly.com/python/plotly-express/) — visualizations  
- [scikit-learn](https://scikit-learn.org/) — ML models & pipelines  
- [openai](https://github.com/openai/openai-python) — GPT-based categorization & insights (optional)  
- [python-dotenv](https://pypi.org/project/python-dotenv/) — manage environment variables  

**Other**
- Built-in SMTP for optional email exports  
- Local heuristic fallbacks (categorization, insights, regret prediction)

---

## ⚙️ Installation & Setup

Clone this repository and set up a local Python environment.

```bash
# Clone the repo
git clone https://github.com/yourusername/losslens.git
cd losslens

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

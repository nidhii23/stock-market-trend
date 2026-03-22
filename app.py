import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

from technical_engine import calculate_technical_score
from fundamental_engine import calculate_fundamental_score
from decision_engine import make_decision
from sentiment import get_news_sentiment
from fundamentals import get_fundamentals
from backtesting import run_backtest, calculate_metrics
from ml_model import prepare_features, train_model, predict_signal

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Stock Prediction System", layout="wide")

st.title("📈 Stock Market Trend Prediction System")
st.subheader("📊 Multi-Stock AI Scanner")

# ==============================
# INPUT
# ==============================
stocks = {
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "L&T": "LT.NS",
    "ITC": "ITC.NS",
    "SBI": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "HCL Tech": "HCLTECH.NS"
}

selected_stock = st.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[selected_stock]

# ==============================
# MAIN LOGIC
# ==============================
if ticker:

    # FETCH DATA
    # 🔥 LONG DATA FOR TRAINING + BACKTEST
 df_train = yf.download(ticker, period="10y", interval="1d", auto_adjust=True)

# 🔥 SHORT DATA FOR CURRENT VIEW
df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        # ==============================
# ML MODEL 🔥
# ==============================
# 🔥 TRAIN ON LONG DATA
df_ml_train = prepare_features(df_train)
model, features, ml_metrics = train_model(df_ml_train)

if model is None:
    ml_signal = "HOLD"
else:
    df_ml = prepare_features(df)
    ml_signal = predict_signal(model, df_ml, features)

# 🔥 PREDICT ON RECENT DATA
df_ml = prepare_features(df)
ml_signal = predict_signal(model, df_ml, features)

    # TECHNICAL
tech = calculate_technical_score(df)

    # FUNDAMENTALS (REAL)
fundamental_data = get_fundamentals(ticker)
fund = calculate_fundamental_score(fundamental_data)

    # SENTIMENT
sentiment = get_news_sentiment(ticker)

# ==============================
# BACKTEST SIGNAL GENERATION 🔥
# ==============================

signals = []

MIN_DATA = 30

for i in range(len(df)):

    temp_df = df.iloc[:i+1]

    # Default signal
    signal = "HOLD"

    if i >= MIN_DATA:

        tech_i = calculate_technical_score(temp_df)
        fund_i = fund
        sent_i = sentiment

        df_ml_i = prepare_features(temp_df)

        if len(df_ml_i) >= 30:
            ml_i = predict_signal(model, df_ml_i, features)

            decision_i = make_decision(tech_i, fund_i, sent_i, ml_i)

            signal = decision_i["signal"]
            

    # Prevent duplicate BUY/SELL
    if len(signals) > 0:
        if signals[-1] == "BUY" and signal == "BUY":
            signal = "HOLD"
        elif signals[-1] == "SELL" and signal == "SELL":
            signal = "HOLD"

    signals.append(signal)


signals = pd.Series(signals, index=df.index)
st.write(pd.Series(signals).value_counts())
# ==============================
# RUN BACKTEST
# ==============================
df_bt, trades = run_backtest(df, signals)
metrics = calculate_metrics(df_bt)

# FINAL DECISION
final = make_decision(tech, fund, sentiment, ml_signal)

    # ==============================
    # UI LAYOUT
    # ==============================
col1, col2 = st.columns(2)

    # PRICE CHART
with col1:
        st.subheader("📊 Price Chart")
       # ==============================
# PRICE CHART WITH SIGNALS 🔥
# ==============================
fig = go.Figure()

# Price line
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    mode="lines",
    name="Price"
))

# BUY signals
buy_signals = df_bt[signals == "BUY"]

fig.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals["Close"],
    mode="markers",
    marker=dict(symbol="triangle-up", size=10),
    name="BUY"
))

# SELL signals
sell_signals = df_bt[signals == "SELL"]

fig.add_trace(go.Scatter(
    x=sell_signals.index,
    y=sell_signals["Close"],
    mode="markers",
    marker=dict(symbol="triangle-down", size=10),
    name="SELL"
))

st.plotly_chart(fig, use_container_width=True)

    # FINAL SIGNAL
with col2:
        st.subheader("🚀 Final Recommendation")
        st.metric("Signal", final["signal"])
        st.metric("Confidence", f"{final['confidence']}%")
        st.metric("Score", final["final_score"])

    # ==============================
    # TECHNICAL
    # ==============================
st.subheader("⚙️ Technical Analysis")

st.write("**Signal:**", tech["signal"])
st.write("**Score:**", tech["score"])

st.write("### ✅ Why BUY")
for r in tech["buy_reasons"]:
        st.write("✔", r)

st.write("### ⚠ Risks")
for r in tech["sell_reasons"]:
        st.write("⚠", r)

    # ==============================
    # FUNDAMENTALS
    # ==============================
st.subheader("📊 Fundamental Analysis")

st.write("**Signal:**", fund["signal"])
st.write("**Score:**", fund["score"])

st.write("### ✅ Why BUY")
for r in fund["buy_reasons"]:
        st.write("✔", r)

st.write("### ⚠ Risks")
for r in fund["sell_reasons"]:
        st.write("⚠", r)

    # ==============================
    # RAW FUNDAMENTALS (NEW)
    # ==============================
st.subheader("📌 Raw Fundamentals")

for k, v in fundamental_data.items():
        st.write(f"{k}: {round(v, 2)}")

    # ==============================
    # SENTIMENT
    # ==============================
st.subheader("📰 News Sentiment")

st.write("**Signal:**", sentiment["signal"])
st.write("**Score:**", sentiment["score"])

st.write("### 🟢 Positive News")
for n in sentiment["positive_news"]:
        st.write("✔", n)

st.write("### 🔴 Negative News")
for n in sentiment["negative_news"]:
        st.write("⚠", n)

    # ==============================
    # FINAL EXPLANATION
    # ==============================
st.subheader("🧠 Final Explanation")

st.write("### ✅ Why BUY")
for r in final["summary"]:
        st.write("✔", r)

st.write("### ⚠ Risks")
for r in final["risks"]:
        st.write("⚠", r)
        # ==============================
# BACKTEST RESULTS UI
# ==============================
st.subheader("📈 Backtesting Results")

st.write("### Performance Metrics")
st.write(metrics)

st.line_chart(df_bt["Portfolio"])

st.write("### Trades")
st.write(trades)
# ==============================
# AI PREDICTION 🔥
# ==============================
st.subheader("🤖 AI Prediction")

st.write("ML Signal:", ml_signal)
st.write("### 📊 ML Performance")
st.write("Accuracy:", f"{ml_metrics['accuracy']}%")
st.write("Precision:", f"{ml_metrics['precision']}%")
import yfinance as yf
import pandas as pd

from technical_engine import calculate_technical_score
from fundamental_engine import calculate_fundamental_score
from decision_engine import make_decision

# ==============================
# FETCH DATA
# ==============================
df = yf.download("TCS.NS", period="6mo", interval="1d", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ==============================
# TECHNICAL
# ==============================
tech_result = calculate_technical_score(df)

# ==============================
# FUNDAMENTAL
# ==============================
fundamental_data = {
    "roe": 22,
    "operating_margin": 18,
    "pe": 25,
    "industry_pe": 30,
    "peg": 0.9,
    "price_to_cash_flow": 20,
    "debt_to_equity": 0.5,
    "sales_growth": 12,
    "profit_growth": 15,
    "promoter_holding": 60
}

fund_result = calculate_fundamental_score(fundamental_data)

# ==============================
# SENTIMENT (TEMP)
# ==============================
sentiment = {
    "score": 0.2,
    "signal": "POSITIVE"
}

# ==============================
# FINAL DECISION
# ==============================
final = make_decision(tech_result, fund_result, sentiment)

print("\nFINAL DECISION")
print("Score:", final["final_score"])
print("Signal:", final["signal"])
print("Confidence:", final["confidence"])

print("\nWHY:")
for r in final["summary"]:
    print("✔", r)

print("\nRISKS:")
for r in final["risks"]:
    print("⚠", r)
    
from sentiment import get_news_sentiment

sentiment = get_news_sentiment("TCS.NS")

print("\nSentiment Score:", sentiment["score"])
print("Signal:", sentiment["signal"])

print("\nPositive News:")
for n in sentiment["positive_news"]:
    print("✔", n)

print("\nNegative News:")
for n in sentiment["negative_news"]:
    print("⚠", n)
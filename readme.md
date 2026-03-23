# AI-Driven Stock Market Trend Prediction System

A multi-signal stock analysis system for NSE-listed Indian equities combining
LSTM+XGBoost machine learning, FinBERT news sentiment, technical analysis,
fundamental scoring, and market regime detection.

## Tested results

| Stock | WF Accuracy | Strategy Return | Sharpe | vs Nifty50 |
|-------|-------------|-----------------|--------|------------|
| SBI   | 59.9%       | +18.51%         | 0.82   | +13.18%    |
| TCS   | 53.3%       | -2.41%          | -0.11  | -7.74%     |

---

## Quick start

### 1. Python version
Use Python **3.10 or 3.11**. Python 3.12 has compatibility issues with nsepy.

```bash
python --version   # should show 3.10.x or 3.11.x
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> First run downloads FinBERT (~400MB) automatically. Subsequent runs load from cache.

### 4. Run the app
```bash
streamlit run app.py
```

Opens at http://localhost:8501

---

## First run time

| Step | Time |
|------|------|
| Install packages | 5–10 min (one time only) |
| FinBERT download | 2–5 min (one time only) |
| First stock analysis | 8–12 min (trains LSTM from scratch) |
| Switching stocks | 8–12 min per new stock |
| Returning to a stock | Instant (cached for 1 hour) |

---

## Optional features

### Enable GNN (Graph Neural Network)
The GNN models correlations between all 10 stocks simultaneously.
Requires torch-geometric which needs a special install matching your torch version.

```bash
# Find your torch version first
python -c "import torch; print(torch.__version__)"

# Then install torch-geometric — replace {TORCH} and {CUDA} with your versions
# CPU only (most laptops):
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{TORCH}+cpu.html
```

Then toggle "Graph Neural Network" in the app sidebar.

### Enable TFT (Temporal Fusion Transformer)
```bash
pip install pytorch-forecasting==1.6.1 lightning>=2.0.0
```

Then toggle "Temporal Fusion Transformer" in the app sidebar.

---

## Troubleshooting

### Yahoo Finance 401 Unauthorized
```bash
pip install yfinance --upgrade
# If still failing, use the last known stable version:
pip install yfinance==0.2.54
```

### FinBERT import error in Streamlit
```bash
pip install transformers==4.40.0
```
The system automatically falls back to VADER sentiment if FinBERT fails.

### PyArrow crash on trade log
Make sure you have the latest `backtesting.py` — older versions had a
mixed-type column issue. Replace backtesting.py from the project files.

### nsepy returns empty data
NSE sometimes blocks requests. The system handles this gracefully —
delivery % defaults to 50% (neutral) if nsepy returns nothing.

### App very slow on first load
This is expected — LSTM trains on 10 years of data on first run.
Subsequent loads of the same stock within 1 hour are instant due to caching.

---

## Project structure

```
stock-market-trend-project/
├── app.py                  # Streamlit UI (rendering only)
├── pipeline.py             # Main orchestrator
├── ml_model.py             # LSTM + XGBoost ensemble
├── decision_engine.py      # Weighted signal combination
├── backtesting.py          # ATR stop-loss backtester
├── benchmark.py            # Nifty50 comparison
├── regime.py               # BULL/BEAR market detector
├── nse_data.py             # PCR, delivery %, sector momentum
├── sentiment.py            # FinBERT + Google News RSS
├── fundamentals.py         # yfinance fundamental fetcher
├── fundamental_engine.py   # Fundamental scoring engine
├── technical_engine.py     # Technical indicator scoring
├── gnn_model.py            # Graph Neural Network (optional)
├── tft_model.py            # Temporal Fusion Transformer (optional)
└── requirements.txt        # This file
```

---

## Covered stocks (26 total)

**Large-cap:** TCS, Reliance, Infosys, HDFC Bank, ICICI Bank, L&T, ITC, SBI, Bharti Airtel, HCL Tech

**Mid-cap IT:** Persistent Systems, Coforge, KPIT Tech, Mphasis

**Mid-cap Banking:** Federal Bank, IDFC First, Bandhan Bank, Cholamandalam

**Mid-cap FMCG:** Tata Consumer, Godrej Consumer, Marico

**Mid-cap Pharma:** Alkem Labs, Ipca Labs, Torrent Pharma

**Mid-cap Auto:** Ashok Leyland, Escorts Kubota, Balkrishna Industries

---

## Understanding the signals

| Score | Signal | Meaning |
|-------|--------|---------|
| > 52% | BUY    | Weighted signals lean bullish — consider entering |
| 45–52%| HOLD   | Mixed signals — wait for clearer direction |
| < 45% | SELL   | Weighted signals lean bearish — avoid or exit |

**Important:** Signals are suppressed to HOLD if:
- Nifty50 is in BEAR market (MA50 < MA200)
- Earnings results are within 14 days
- ML confidence is below 55%

---

## Not financial advice
This system is for educational and research purposes only.
Always do your own research before making any investment decisions.
Past backtest performance does not guarantee future results.
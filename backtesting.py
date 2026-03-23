"""
backtesting.py — Long-only backtester.

Changes:
  - ATR-based dynamic stop-loss (replaces fixed 3%)
  - Position sizing based on ML confidence
  - Nifty50 benchmark comparison column
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INITIAL_CASH = 100_000.0
ATR_MULTIPLIER = 2.0     # stop-loss = entry - 2x ATR
ATR_WINDOW     = 14
TAKE_PROFIT    = 0.08    # 8% take-profit (raised from 5%)
MIN_POSITION   = 0.5     # minimum 50% of cash per trade
MAX_POSITION   = 1.0     # maximum 100% of cash per trade


def _atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    """Average True Range."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"].shift(1)
    tr    = pd.concat([
        high - low,
        (high - close).abs(),
        (low  - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def run_backtest(
    df:           pd.DataFrame,
    signals:      pd.Series,
    confidences:  pd.Series | None = None,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a long-only strategy.

    Parameters
    ----------
    df          : OHLCV DataFrame with Close, High, Low
    signals     : pd.Series of 'BUY'/'SELL'/'HOLD'
    confidences : optional pd.Series of ML confidence (0.5–1.0) for position sizing
    initial_cash: starting capital

    Returns
    -------
    (df_out, trades_df)
    """
    df = df.copy()

    # Compute ATR for dynamic stop-loss
    if "High" in df.columns and "Low" in df.columns:
        df["_atr"] = _atr(df)
    else:
        df["_atr"] = df["Close"] * 0.02   # fallback: 2% of price

    cash        = initial_cash
    shares      = 0.0
    position    = False
    entry_price = 0.0
    stop_price  = 0.0

    portfolio_values: list[float] = []
    trades:           list[dict]  = []

    for i in range(len(df)):
        price  = float(df["Close"].iloc[i])
        signal = signals.iloc[i]
        atr    = float(df["_atr"].iloc[i]) if pd.notna(df["_atr"].iloc[i]) else price * 0.02

        # ── Entry ────────────────────────────────────────
        if signal == "BUY" and not position:
            # Position sizing: scale by confidence (0.5–1.0 → 50–100% of cash)
            conf = float(confidences.iloc[i]) if confidences is not None else 0.75
            pos_size = MIN_POSITION + (MAX_POSITION - MIN_POSITION) * ((conf - 0.5) / 0.5)
            pos_size = max(MIN_POSITION, min(MAX_POSITION, pos_size))

            invest      = cash * pos_size
            shares      = invest / price
            cash       -= invest
            position    = True
            entry_price = price
            # Dynamic ATR-based stop-loss
            stop_price  = entry_price - (ATR_MULTIPLIER * atr)

            trades.append({
                "action":    "BUY",
                "date":      df.index[i],
                "price":     round(price, 4),
                "shares":    round(shares, 4),
                "value":     round(shares * price, 2),
                "pnl":       0.0,
                "stop":      round(stop_price, 4),
                "pos_size":  round(pos_size * 100, 1),
            })

        # ── Exit ─────────────────────────────────────────
        elif position:
            hit_stop   = price < stop_price
            hit_target = price > entry_price * (1 + TAKE_PROFIT)
            exit_now   = signal == "SELL" or hit_stop or hit_target

            if exit_now:
                value  = shares * price
                pnl    = value - (shares * entry_price)
                reason = ("STOP-LOSS"   if hit_stop   else
                          "TAKE-PROFIT" if hit_target else "SIGNAL")
                trades.append({
                    "action":   f"SELL ({reason})",
                    "date":     df.index[i],
                    "price":    round(price, 4),
                    "shares":   round(shares, 4),
                    "value":    round(value, 2),
                    "pnl":      round(pnl, 2),
                    "stop":     round(stop_price, 4),
                    "pos_size": None,
                })
                cash    += value
                shares   = 0.0
                position = False

        portfolio_values.append(shares * price + cash)  # mark-to-market (includes open positions)

    df["Portfolio"] = portfolio_values
    df.drop(columns=["_atr"], inplace=True, errors="ignore")

    # Mark any still-open position as unrealised in trade log
    if position and shares > 0:
        last_price = float(df["Close"].iloc[-1])
        unreal_pnl = (last_price - entry_price) * shares
        trades.append({
            "action":   "OPEN (unrealised)",
            "date":     df.index[-1],
            "price":    round(last_price, 4),
            "shares":   round(shares, 4),
            "value":    round(shares * last_price, 2),
            "pnl":      round(unreal_pnl, 2),
            "stop":     round(stop_price, 4),
            "pos_size": None,
        })

    trades_df = (pd.DataFrame(trades) if trades
                 else pd.DataFrame(columns=[
                     "action","date","price","shares","value","pnl","stop","pos_size"]))
    # Cast pos_size to float so PyArrow can serialise it (None → NaN, not "")
    if "pos_size" in trades_df.columns:
        trades_df["pos_size"] = pd.to_numeric(trades_df["pos_size"], errors="coerce")
    return df, trades_df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Performance metrics from backtested portfolio."""
    port = df["Portfolio"].dropna()

    if len(port) < 2:
        return {"error": "Insufficient data for metrics."}

    total_return  = (port.iloc[-1] / port.iloc[0] - 1) * 100
    daily_returns = port.pct_change().dropna()
    sharpe        = (float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
                     if daily_returns.std() != 0 else 0.0)
    peak          = port.cummax()
    max_dd        = float(((port - peak) / peak).min()) * 100
    n_years       = len(port) / 252
    cagr          = ((port.iloc[-1] / port.iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    win_rate      = float((daily_returns > 0).mean()) * 100

    return {
        "Total Return (%)": round(total_return, 2),
        "CAGR (%)":          round(cagr, 2),
        "Sharpe Ratio":      round(sharpe, 2),
        "Max Drawdown (%)":  round(max_dd, 2),
        "Win Rate (%)":      round(win_rate, 2),
        "Trading Days":      len(port),
    }
"""
technical_engine.py — Technical indicator scoring.

New: India VIX fear indicator + USD/INR rate (critical for IT stocks).
"""

import logging
import pandas as pd
import yfinance as yf
import ta

logger = logging.getLogger(__name__)

_W = dict(
    ema_trend  = 0.20,
    macd       = 0.20,
    rsi        = 0.15,
    bb         = 0.15,
    volume     = 0.10,
    support    = 0.10,
    adx        = 0.10,
)

_VIX_TICKER   = "^INDIAVIX"
_USDINR_TICKER = "INR=X"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _fetch_vix() -> float | None:
    """Fetch latest India VIX value."""
    try:
        df = yf.download(_VIX_TICKER, period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].dropna().iloc[-1]) if not df.empty else None
    except Exception:
        return None


def _fetch_usdinr() -> float | None:
    """Fetch latest USD/INR rate."""
    try:
        df = yf.download(_USDINR_TICKER, period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return float(df["Close"].dropna().iloc[-1]) if not df.empty else None
    except Exception:
        return None


def calculate_technical_score(df: pd.DataFrame,
                               fetch_macro: bool = False) -> dict:
    """
    Score a stock 0–1 based on technical indicators.

    Parameters
    ----------
    df           : OHLCV DataFrame (needs Close, High, Low, Volume)
    fetch_macro  : if True, also fetch India VIX and USD/INR (slower)

    Returns
    -------
    dict: score, signal, buy_reasons, sell_reasons, vix, usdinr
    """
    df = _flatten(df).copy()

    required = {"Close", "High", "Low", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"technical_engine: missing columns {missing}")

    if len(df) < 30:
        logger.warning("calculate_technical_score: only %d rows.", len(df))

    # ── Indicators ──────────────────────────────────────
    df["ema_200"]     = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()
    df["rsi"]         = ta.momentum.RSIIndicator(df["Close"]).rsi()
    _macd             = ta.trend.MACD(df["Close"])
    df["macd"]        = _macd.macd()
    df["macd_signal"] = _macd.macd_signal()
    _bb               = ta.volatility.BollingerBands(df["Close"])
    df["bb_high"]     = _bb.bollinger_hband()
    df["bb_low"]      = _bb.bollinger_lband()
    df["adx"]         = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"]).adx()

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    score        = 0.0
    buy_reasons  = []
    sell_reasons = []

    # ── EMA trend ────────────────────────────────────────
    if pd.notna(latest["ema_200"]):
        if latest["Close"] > latest["ema_200"]:
            score += _W["ema_trend"]
            buy_reasons.append("Price above 200 EMA → uptrend confirmed")
        else:
            sell_reasons.append("Price below 200 EMA → downtrend")

    # ── MACD ─────────────────────────────────────────────
    if pd.notna(latest["macd"]) and pd.notna(latest["macd_signal"]):
        if latest["macd"] > latest["macd_signal"]:
            score += _W["macd"]
            buy_reasons.append("MACD above signal → bullish momentum")
        else:
            sell_reasons.append("MACD below signal → bearish momentum")

    # ── RSI ──────────────────────────────────────────────
    if pd.notna(latest["rsi"]):
        if latest["rsi"] < 30:
            score += _W["rsi"]
            buy_reasons.append(f"RSI oversold ({latest['rsi']:.0f}) → potential reversal")
        elif latest["rsi"] > 70:
            sell_reasons.append(f"RSI overbought ({latest['rsi']:.0f}) → caution")
        else:
            score += _W["rsi"] * 0.5

    # ── Bollinger Bands ───────────────────────────────────
    if pd.notna(latest["bb_low"]) and pd.notna(latest["bb_high"]):
        if latest["Close"] <= latest["bb_low"]:
            score += _W["bb"]
            buy_reasons.append("Price at lower Bollinger Band → oversold")
        elif latest["Close"] >= latest["bb_high"]:
            sell_reasons.append("Price at upper Bollinger Band → overbought")

    # ── Volume ───────────────────────────────────────────
    if pd.notna(prev["Volume"]) and prev["Volume"] > 0:
        if latest["Volume"] > prev["Volume"]:
            if latest["Close"] > prev["Close"]:
                score += _W["volume"]
                buy_reasons.append("Rising volume on up day → accumulation")
            else:
                sell_reasons.append("Rising volume on down day → distribution")

    # ── Support / resistance ──────────────────────────────
    support    = df["Low"].rolling(20).min().iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]
    if pd.notna(support) and latest["Close"] <= support * 1.02:
        score += _W["support"]
        buy_reasons.append("Price near 20-day support")
    elif pd.notna(resistance) and latest["Close"] >= resistance * 0.98:
        sell_reasons.append("Price near 20-day resistance")

    # ── ADX ──────────────────────────────────────────────
    if pd.notna(latest["adx"]) and latest["adx"] > 25:
        score += _W["adx"]
        buy_reasons.append(f"Strong trend (ADX {latest['adx']:.0f} > 25)")

    # ── Macro overlays ───────────────────────────────────
    vix    = None
    usdinr = None

    if fetch_macro:
        vix    = _fetch_vix()
        usdinr = _fetch_usdinr()

        if vix is not None:
            if vix > 20:
                sell_reasons.append(f"India VIX {vix:.1f} > 20 → high fear, risk-off")
            elif vix < 13:
                buy_reasons.append(f"India VIX {vix:.1f} < 13 → low fear, risk-on")

        if usdinr is not None:
            if usdinr > 84:
                sell_reasons.append(f"USD/INR {usdinr:.1f} → weak rupee, IT margins at risk")
            elif usdinr < 82:
                buy_reasons.append(f"USD/INR {usdinr:.1f} → strong rupee, import cost benefit")

    score = round(min(score, 1.0), 4)

    if score > 0.70:   signal = "STRONG BUY"
    elif score > 0.50: signal = "BUY"
    elif score > 0.30: signal = "HOLD"
    else:              signal = "SELL"

    return {
        "score":        score,
        "signal":       signal,
        "buy_reasons":  buy_reasons,
        "sell_reasons": sell_reasons,
        "vix":          vix,
        "usdinr":       usdinr,
    }
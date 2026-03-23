"""
regime.py — Market regime detector using Nifty50.

Detects whether the broad Indian market is in a BULL or BEAR regime
using MA50 vs MA200 (Golden Cross / Death Cross logic).

Used by decision_engine.py to suppress BUY signals in bear markets.
"""

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

NIFTY_TICKER = "^NSEI"
_cache: dict = {}


def get_market_regime(force_refresh: bool = False) -> dict:
    """
    Detect current Nifty50 market regime.

    Returns
    -------
    dict with keys:
        regime      — 'BULL' | 'BEAR' | 'UNKNOWN'
        ma50        — 50-day moving average
        ma200       — 200-day moving average
        last_close  — latest Nifty50 close
        signal      — human readable string
    """
    if not force_refresh and _cache:
        return _cache.copy()

    try:
        df = yf.download(NIFTY_TICKER, period="1y", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty or len(df) < 50:
            return _unknown()

        close    = df["Close"]
        ma50     = float(close.rolling(50).mean().iloc[-1])
        ma200    = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else ma50
        last     = float(close.iloc[-1])

        regime   = "BULL" if ma50 > ma200 else "BEAR"
        gap_pct  = round((ma50 - ma200) / ma200 * 100, 2) if ma200 else 0

        signal = (
            f"Nifty50 MA50 ({ma50:.0f}) {'>' if regime=='BULL' else '<'} "
            f"MA200 ({ma200:.0f}) — {regime} market"
        )

        result = {
            "regime":     regime,
            "ma50":       round(ma50, 2),
            "ma200":      round(ma200, 2),
            "last_close": round(last, 2),
            "gap_pct":    gap_pct,
            "signal":     signal,
        }
        _cache.update(result)
        logger.info("Market regime: %s (MA50=%.0f MA200=%.0f)", regime, ma50, ma200)
        return result

    except Exception as exc:
        logger.warning("get_market_regime failed: %s", exc)
        return _unknown()


def _unknown() -> dict:
    return {
        "regime": "UNKNOWN", "ma50": 0, "ma200": 0,
        "last_close": 0, "gap_pct": 0,
        "signal": "Could not determine market regime",
    }
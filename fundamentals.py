"""
fundamentals.py — Fetch raw fundamental data from yfinance.

Fix: _raw() returns None for missing fields (not 0).
     info is checked to be a dict before use.
"""

import logging
import yfinance as yf

logger = logging.getLogger(__name__)


def _pct(info: dict, key: str) -> float | None:
    v = info.get(key)
    if v is None:
        return None
    try:
        return float(v) * 100
    except (TypeError, ValueError):
        return None


def _raw(info: dict, key: str) -> float | None:
    v = info.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def get_fundamentals(ticker: str) -> dict:
    """
    Fetch fundamental metrics for *ticker* via yfinance.
    All values are float or None — None means data unavailable.
    """
    try:
        info = yf.Ticker(ticker).info
        # yfinance sometimes returns None or non-dict on auth errors
        if not isinstance(info, dict):
            logger.warning("get_fundamentals(%s): info is not a dict (%s)", ticker, type(info))
            info = {}
    except Exception as exc:
        logger.warning("get_fundamentals(%s): fetch failed — %s", ticker, exc)
        info = {}

    d2e_raw = _raw(info, "debtToEquity")

    return {
        "roe":               _pct(info, "returnOnEquity"),
        "operating_margin":  _pct(info, "operatingMargins"),
        "pe":                _raw(info, "trailingPE"),
        "industry_pe":       _raw(info, "forwardPE"),
        "peg":               _raw(info, "pegRatio"),
        "price_to_cash_flow": _raw(info, "priceToCashflow"),
        "debt_to_equity":    round(d2e_raw / 100, 4) if d2e_raw is not None else None,
        "sales_growth":      _pct(info, "revenueGrowth"),
        "profit_growth":     _pct(info, "earningsGrowth"),
        "promoter_holding":  _pct(info, "heldPercentInsiders"),
    }
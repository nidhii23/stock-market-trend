"""
nse_data.py — NSE-specific data fetchers.

Provides:
  1. Delivery % — genuine buying vs speculation (from nsepy)
  2. Put/Call ratio — options market fear indicator (from NSE website)
  3. Sector ETF momentum — is the whole sector moving? (from yfinance)
  4. Earnings calendar — when are results due? (from yfinance)

All functions fail gracefully — returns None/empty if data unavailable.
Install: pip install nsepy requests
"""

import logging
import warnings
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd
import numpy as np
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Sector ETF mapping ────────────────────────────────────
# Maps stock ticker suffix to sector proxy ticker
SECTOR_MAP = {
    # IT stocks
    "TCS.NS":        "INFY.NS",        # use Infosys as IT proxy
    "INFY.NS":       "TCS.NS",
    "HCLTECH.NS":    "TCS.NS",
    "WIPRO.NS":      "TCS.NS",
    "PERSISTENT.NS": "TCS.NS",
    "COFORGE.NS":    "TCS.NS",
    "KPITTECH.NS":   "TCS.NS",
    "MPHASIS.NS":    "TCS.NS",
    # Banking stocks
    "HDFCBANK.NS":   "SBIN.NS",        # use SBI as banking proxy
    "ICICIBANK.NS":  "HDFCBANK.NS",
    "SBIN.NS":       "HDFCBANK.NS",
    "FEDERALBNK.NS": "HDFCBANK.NS",
    "IDFCFIRSTB.NS": "HDFCBANK.NS",
    "BANDHANBNK.NS": "HDFCBANK.NS",
    # FMCG
    "ITC.NS":        "HINDUNILVR.NS",
    "TATACONSUM.NS": "ITC.NS",
    "GODREJCP.NS":   "ITC.NS",
    "MARICO.NS":     "ITC.NS",
    # Others
    "RELIANCE.NS":   "LT.NS",
    "LT.NS":         "RELIANCE.NS",
    "BHARTIARTL.NS": "RELIANCE.NS",
}

# Earnings month map for Indian companies (approximate Q results months)
# Q1 = Apr-Jun results in Jul-Aug, Q2 = Jul-Sep results in Oct-Nov
# Q3 = Oct-Dec results in Jan-Feb, Q4 = Jan-Mar results in Apr-May
EARNINGS_MONTHS = {
    "TCS.NS":        [1, 4, 7, 10],   # Jan, Apr, Jul, Oct
    "INFY.NS":       [1, 4, 7, 10],
    "HCLTECH.NS":    [1, 4, 7, 10],
    "WIPRO.NS":      [1, 4, 7, 10],
    "PERSISTENT.NS": [1, 4, 7, 10],
    "COFORGE.NS":    [1, 4, 7, 10],
    "HDFCBANK.NS":   [1, 4, 7, 10],
    "ICICIBANK.NS":  [1, 4, 7, 10],
    "SBIN.NS":       [2, 5, 8, 11],   # Feb, May, Aug, Nov
    "RELIANCE.NS":   [1, 4, 7, 10],
    "ITC.NS":        [2, 5, 8, 11],
    "LT.NS":         [2, 5, 8, 11],
    "BHARTIARTL.NS": [2, 5, 8, 11],
}


# ══════════════════════════════════════════════
# 1. DELIVERY PERCENTAGE
# ══════════════════════════════════════════════

def get_delivery_pct(
    symbol: str,
    start: date,
    end: date,
) -> Optional[pd.Series]:
    """
    Fetch delivery % from NSE via nsepy.

    Delivery % = (Deliverable Volume / Total Volume) × 100
    High delivery % (>50%) = genuine long-term buying, not speculation.
    Low delivery % (<25%) = mostly intraday/speculative trading.

    Parameters
    ----------
    symbol : NSE symbol without .NS suffix (e.g. 'TCS', 'SBIN')
    start, end : date range

    Returns
    -------
    pd.Series indexed by date, values 0-100, or None if unavailable.
    """
    # Strip .NS suffix if present
    sym = symbol.replace(".NS", "").replace(".BO", "").upper()

    try:
        from nsepy import get_history
        df = get_history(symbol=sym, start=start, end=end)

        if df.empty:
            logger.warning("get_delivery_pct(%s): empty response from NSE", sym)
            return None

        if "%Deliverble" in df.columns:
            series = df["%Deliverble"].dropna() * 100   # convert 0-1 to 0-100
            logger.info("get_delivery_pct(%s): %d rows", sym, len(series))
            return series
        elif "Deliverable Volume" in df.columns and "Volume" in df.columns:
            series = (df["Deliverable Volume"] / df["Volume"].replace(0, np.nan)) * 100
            return series.dropna()
        else:
            logger.warning("get_delivery_pct(%s): delivery columns not found", sym)
            return None

    except ImportError:
        logger.warning("nsepy not installed. pip install nsepy")
        return None
    except Exception as exc:
        logger.warning("get_delivery_pct(%s): %s", sym, exc)
        return None


def get_delivery_features(
    ticker: str,
    df_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Return a DataFrame with delivery % features aligned to df_index.

    Features:
        delivery_pct      — raw delivery %
        delivery_ma10     — 10-day MA of delivery %
        delivery_signal   — delivery % vs its 20-day average (z-score)
    """
    sym   = ticker.replace(".NS", "").upper()
    start = df_index[0].date() - timedelta(days=60)  # buffer for MA
    end   = df_index[-1].date()

    result = pd.DataFrame(index=df_index)
    result["delivery_pct"]    = np.nan
    result["delivery_ma10"]   = np.nan
    result["delivery_signal"] = 0.0

    series = get_delivery_pct(sym, start, end)
    if series is None or len(series) < 5:
        return result

    series.index = pd.to_datetime(series.index)
    series       = series.reindex(df_index, method="ffill")

    result["delivery_pct"]    = series
    result["delivery_ma10"]   = series.rolling(10).mean()
    # Normalised signal: how unusual is today's delivery vs recent history?
    rolling_mean = series.rolling(20).mean()
    rolling_std  = series.rolling(20).std().replace(0, np.nan)
    result["delivery_signal"] = ((series - rolling_mean) / rolling_std).clip(-3, 3)

    return result


# ══════════════════════════════════════════════
# 2. PUT/CALL RATIO (PCR)
# ══════════════════════════════════════════════

def get_pcr_nifty() -> Optional[float]:
    """
    Fetch current Nifty50 Put/Call Ratio from NSE website.

    PCR > 1.2 → more puts than calls → bearish sentiment / fear
    PCR < 0.8 → more calls than puts → bullish / greed
    PCR 0.8-1.2 → neutral

    Returns current PCR float or None if unavailable.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com",
        }
        session = requests.Session()
        # First hit the main page to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        resp = session.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Sum all PE and CE OI across strikes
        records = data.get("records", {}).get("data", [])
        total_ce_oi = sum(
            r.get("CE", {}).get("openInterest", 0) for r in records if "CE" in r
        )
        total_pe_oi = sum(
            r.get("PE", {}).get("openInterest", 0) for r in records if "PE" in r
        )

        if total_ce_oi == 0:
            return None

        pcr = round(total_pe_oi / total_ce_oi, 3)
        logger.info("Nifty PCR: %.3f", pcr)
        return pcr

    except Exception as exc:
        logger.warning("get_pcr_nifty: %s", exc)
        return None


def pcr_to_signal(pcr: Optional[float]) -> dict:
    """
    Convert PCR value to a sentiment signal.

    Returns dict with pcr, signal, normalised score [0,1].
    """
    if pcr is None:
        return {"pcr": None, "signal": "NEUTRAL", "score": 0.5}

    if pcr > 1.5:
        signal = "EXTREME FEAR"
        score  = 0.2    # contrarian BUY signal — market too pessimistic
    elif pcr > 1.2:
        signal = "BEARISH"
        score  = 0.35
    elif pcr > 0.9:
        signal = "NEUTRAL"
        score  = 0.5
    elif pcr > 0.7:
        signal = "BULLISH"
        score  = 0.65
    else:
        signal = "EXTREME GREED"
        score  = 0.8    # contrarian SELL — market too optimistic

    return {"pcr": pcr, "signal": signal, "score": score}


# ══════════════════════════════════════════════
# 3. SECTOR ETF MOMENTUM
# ══════════════════════════════════════════════

@lru_cache(maxsize=20)
def _fetch_sector_data(sector_ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """Cached sector price data fetch."""
    try:
        df = yf.download(sector_ticker, period=period,
                         interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except Exception:
        return None


def get_sector_momentum(ticker: str) -> dict:
    """
    Calculate sector momentum for a stock's sector.

    Uses correlated large-cap as sector proxy (see SECTOR_MAP).
    Returns 5-day and 21-day sector momentum + relative strength.

    Returns dict with:
        sector_5d     — sector 5-day return
        sector_21d    — sector 21-day return
        sector_signal — 'STRONG' | 'MODERATE' | 'WEAK' | 'UNKNOWN'
        score         — [0,1] normalised
    """
    proxy = SECTOR_MAP.get(ticker)
    if not proxy:
        return {"sector_5d": 0, "sector_21d": 0,
                "sector_signal": "UNKNOWN", "score": 0.5}

    df = _fetch_sector_data(proxy)
    if df is None or len(df) < 25:
        return {"sector_5d": 0, "sector_21d": 0,
                "sector_signal": "UNKNOWN", "score": 0.5}

    close  = df["Close"]
    ret_5  = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0
    ret_21 = float((close.iloc[-1] / close.iloc[-22] - 1) * 100) if len(close) >= 22 else 0

    # Score based on both timeframes
    score = 0.5
    score += min(0.25, max(-0.25, ret_5  / 20))   # ±5% → ±0.25
    score += min(0.25, max(-0.25, ret_21 / 40))   # ±10% → ±0.25
    score = round(max(0.0, min(1.0, score)), 4)

    if score > 0.65:   signal = "STRONG"
    elif score > 0.50: signal = "MODERATE"
    elif score > 0.35: signal = "WEAK"
    else:              signal = "VERY WEAK"

    return {
        "sector_5d":     round(ret_5, 2),
        "sector_21d":    round(ret_21, 2),
        "sector_signal": signal,
        "score":         score,
        "proxy_ticker":  proxy,
    }


# ══════════════════════════════════════════════
# 4. EARNINGS CALENDAR
# ══════════════════════════════════════════════

def get_earnings_proximity(ticker: str, target_date: Optional[date] = None) -> dict:
    """
    Estimate how close the current date is to the next earnings date.

    Uses known quarterly result months for Indian companies.
    Returns a risk score: high risk = within 2 weeks of results.

    Returns dict with:
        days_to_earnings  — estimated days to next results
        earnings_risk     — 'HIGH' | 'MEDIUM' | 'LOW'
        score             — [0,1] risk score (1 = avoid trading)
        next_month        — estimated month of next results
    """
    if target_date is None:
        target_date = date.today()

    # Try yfinance first
    try:
        info = yf.Ticker(ticker).calendar
        if info is not None and hasattr(info, 'get'):
            earnings_date = info.get("Earnings Date")
            if earnings_date:
                if hasattr(earnings_date, '__iter__'):
                    earnings_date = list(earnings_date)[0]
                if hasattr(earnings_date, 'date'):
                    earnings_date = earnings_date.date()
                days_left = (earnings_date - target_date).days
                if 0 <= days_left <= 60:
                    risk  = "HIGH" if days_left <= 14 else ("MEDIUM" if days_left <= 30 else "LOW")
                    score = max(0.0, 1.0 - days_left / 60)
                    return {
                        "days_to_earnings": days_left,
                        "earnings_risk":    risk,
                        "score":            round(score, 3),
                        "source":           "yfinance",
                    }
    except Exception:
        pass

    # Fallback: estimate from known results months
    months = EARNINGS_MONTHS.get(ticker, [1, 4, 7, 10])
    current_month = target_date.month

    # Find next results month
    next_month = None
    min_days   = 999
    for m in months:
        # Try this year and next year
        for year_offset in [0, 1]:
            year = target_date.year + year_offset
            try:
                results_date = date(year, m, 15)   # approximate mid-month
                days_left    = (results_date - target_date).days
                if 0 <= days_left < min_days:
                    min_days   = days_left
                    next_month = m
            except ValueError:
                pass

    if min_days == 999:
        return {"days_to_earnings": None, "earnings_risk": "UNKNOWN",
                "score": 0.0, "source": "estimate"}

    risk  = "HIGH" if min_days <= 14 else ("MEDIUM" if min_days <= 30 else "LOW")
    score = max(0.0, 1.0 - min_days / 60)

    return {
        "days_to_earnings": min_days,
        "earnings_risk":    risk,
        "score":            round(score, 3),
        "next_month":       next_month,
        "source":           "estimate",
    }


# ══════════════════════════════════════════════
# 5. COMBINED FEATURE VECTOR
# ══════════════════════════════════════════════

def get_all_nse_features(ticker: str) -> dict:
    """
    Fetch all NSE-specific features for a ticker in one call.

    Returns a flat dict used by decision_engine and displayed in app.
    """
    pcr_data      = pcr_to_signal(get_pcr_nifty())
    sector_data   = get_sector_momentum(ticker)
    earnings_data = get_earnings_proximity(ticker)

    return {
        # PCR
        "pcr":              pcr_data.get("pcr"),
        "pcr_signal":       pcr_data.get("signal", "NEUTRAL"),
        "pcr_score":        pcr_data.get("score", 0.5),

        # Sector
        "sector_5d":        sector_data.get("sector_5d", 0),
        "sector_21d":       sector_data.get("sector_21d", 0),
        "sector_signal":    sector_data.get("sector_signal", "UNKNOWN"),
        "sector_score":     sector_data.get("score", 0.5),

        # Earnings
        "days_to_earnings": earnings_data.get("days_to_earnings"),
        "earnings_risk":    earnings_data.get("earnings_risk", "UNKNOWN"),
        "earnings_score":   earnings_data.get("score", 0.0),
    }
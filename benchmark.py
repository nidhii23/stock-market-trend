"""
benchmark.py — Nifty50 buy-and-hold benchmark comparison.

Compares your strategy's performance against simply buying
and holding the Nifty50 index for the same period.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

NIFTY_TICKER  = "^NSEI"
INITIAL_CASH  = 100_000.0


def get_nifty_benchmark(start_date, end_date=None) -> dict:
    """
    Calculate Nifty50 buy-and-hold performance for a given date range.

    Parameters
    ----------
    start_date : str or datetime — start of period
    end_date   : str or datetime — end of period (default: today)

    Returns
    -------
    dict with keys:
        total_return_pct  — total return %
        cagr_pct          — annualised return %
        sharpe            — Sharpe ratio
        max_drawdown_pct  — max drawdown %
        series            — pd.Series of portfolio values (indexed by date)
    """
    try:
        df = yf.download(
            NIFTY_TICKER,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            return _empty_benchmark()

        close    = df["Close"].dropna()
        shares   = INITIAL_CASH / float(close.iloc[0])
        series   = close * shares

        returns  = series.pct_change().dropna()
        total_r  = (series.iloc[-1] / series.iloc[0] - 1) * 100
        n_years  = len(series) / 252
        cagr     = ((series.iloc[-1] / series.iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
        sharpe   = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() != 0 else 0
        peak     = series.cummax()
        dd       = (series - peak) / peak
        max_dd   = float(dd.min()) * 100

        return {
            "total_return_pct": round(total_r, 2),
            "cagr_pct":         round(cagr, 2),
            "sharpe":           round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "series":           series,
        }

    except Exception as exc:
        logger.warning("get_nifty_benchmark failed: %s", exc)
        return _empty_benchmark()


def compare_to_benchmark(strategy_metrics: dict, benchmark: dict) -> dict:
    """
    Compare strategy metrics against benchmark and return alpha/beta summary.

    Returns
    -------
    dict with outperformance on each metric and an overall verdict.
    """
    s, b = strategy_metrics, benchmark

    ret_alpha  = round(s.get("Total Return (%)", 0)  - b.get("total_return_pct", 0), 2)
    cagr_alpha = round(s.get("CAGR (%)", 0)           - b.get("cagr_pct", 0), 2)
    sharpe_diff= round(s.get("Sharpe Ratio", 0)       - b.get("sharpe", 0), 2)

    verdict = "OUTPERFORMING" if ret_alpha > 0 and sharpe_diff > 0 else (
              "UNDERPERFORMING" if ret_alpha < 0 else "MIXED")

    return {
        "return_alpha":      ret_alpha,
        "cagr_alpha":        cagr_alpha,
        "sharpe_difference": sharpe_diff,
        "verdict":           verdict,
        "strategy_return":   s.get("Total Return (%)", 0),
        "benchmark_return":  b.get("total_return_pct", 0),
        "strategy_sharpe":   s.get("Sharpe Ratio", 0),
        "benchmark_sharpe":  b.get("sharpe", 0),
    }


def _empty_benchmark() -> dict:
    return {
        "total_return_pct": 0, "cagr_pct": 0, "sharpe": 0,
        "max_drawdown_pct": 0, "series": pd.Series(dtype=float),
    }
"""
pipeline.py — Simple pipeline. Trains fresh, caches in Streamlit session.

No disk cache, no scheduler, no model files.
Streamlit's @st.cache_data handles caching within a session.

To use:
    from pipeline import run_pipeline, INITIAL_CASH
    r = run_pipeline(ticker, use_tft=False, use_gnn=False)
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

INITIAL_CASH = 100_000.0
SEQUENCE_LEN = 40
MIN_WINDOW   = 30


# ──────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────

@dataclass
class PipelineResult:
    final:            dict
    tech:             dict
    fund:             dict
    sentiment:        dict
    regime:           dict
    nse:              dict
    ml_result:        object
    ml_signal:        str
    ml_confidence:    float
    tft_result:       Optional[object]
    tft_signal:       str
    tft_confidence:   float
    gnn_result:       Optional[object]
    gnn_signal:       str
    gnn_confidence:   float
    fundamental_data: dict
    df_short:         pd.DataFrame
    df_bt:            pd.DataFrame
    signals:          pd.Series
    confidences:      pd.Series
    hist_regime:      pd.Series
    trades:           pd.DataFrame
    metrics:          dict
    benchmark:        dict
    comparison:       dict
    cache_status:     str = "trained_live"
    error:            Optional[str] = None


def _empty(error: str) -> PipelineResult:
    e = pd.DataFrame()
    s = pd.Series(dtype=str)
    return PipelineResult(
        final={}, tech={}, fund={}, sentiment={}, regime={}, nse={},
        ml_result=None, ml_signal="NO_SIGNAL", ml_confidence=0.5,
        tft_result=None, tft_signal="NO_SIGNAL", tft_confidence=0.5,
        gnn_result=None, gnn_signal="NO_SIGNAL", gnn_confidence=0.5,
        fundamental_data={}, df_short=e, df_bt=e,
        signals=s, confidences=s, hist_regime=s,
        trades=e, metrics={}, benchmark={}, comparison={},
        error=error,
    )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _download(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d",
                     auto_adjust=True, progress=False)
    return _flatten(df)


def _hist_regime(df_long: pd.DataFrame, df_short: pd.DataFrame) -> pd.Series:
    """Per-date Nifty50 BULL/BEAR aligned to df_short index."""
    try:
        nifty = _flatten(yf.download(
            "^NSEI", start=df_long.index[0], end=df_short.index[-1],
            interval="1d", auto_adjust=True, progress=False,
        ))
        ma50  = nifty["Close"].rolling(50).mean()
        ma200 = nifty["Close"].rolling(200).mean()
        return (ma50 > ma200).reindex(
            df_short.index, method="ffill").fillna(False)
    except Exception:
        return pd.Series(True, index=df_short.index)


def _build_signals(
    df_long, df_short, hist_regime,
    ml_result, df_feat, fund, nse,
) -> tuple[pd.Series, pd.Series]:
    """Vectorised backtest signal generation."""
    from ml_model import predict_signal, SEQUENCE_LEN as SEQ
    from technical_engine import calculate_technical_score
    from decision_engine import make_decision

    df_feat_bt     = df_feat[df_feat.index.isin(df_short.index)]
    ml_conf_s      = pd.Series(0.5,         index=df_short.index)
    ml_sig_s       = pd.Series("NO_SIGNAL", index=df_short.index)

    if ml_result.is_valid:
        for idx in df_feat_bt.index:
            try:
                rows = df_feat[df_feat.index <= idx]
                if len(rows) >= SEQ:
                    s, c = predict_signal(ml_result, rows)
                    ml_sig_s[idx]  = s
                    ml_conf_s[idx] = c
            except Exception:
                pass

    # Technical scores every 5 days (stride for speed)
    tech_scores: dict = {}
    for date in df_short.index[MIN_WINDOW::5]:
        try:
            w = df_long[df_long.index <= date]
            if len(w) >= 30:
                tech_scores[date] = calculate_technical_score(w)
        except Exception:
            pass

    neutral_sent = {
        "score": 0, "normalised": 0.5, "signal": "NEUTRAL",
        "positive_news": [], "negative_news": [],
    }

    signals_list, conf_list = [], []
    for i, date in enumerate(df_short.index):
        signal = "HOLD"
        conf   = float(ml_conf_s.get(date, 0.5))
        date_regime = "BULL" if hist_regime.iloc[i] else "BEAR"

        if i >= MIN_WINDOW:
            try:
                past = [d for d in tech_scores if d <= date]
                t_i  = tech_scores[past[-1]] if past else None
                if t_i:
                    ml_i   = ml_sig_s.get(date, "NO_SIGNAL")
                    conf_i = float(ml_conf_s.get(date, 0.5))
                    dec_i  = make_decision(
                        t_i, fund, neutral_sent, ml_i, conf_i,
                        market_regime=date_regime,
                        nse_features=nse,
                    )
                    signal = dec_i["signal"]
                    conf   = conf_i
            except Exception:
                pass

        if signals_list:
            last = signals_list[-1]
            if last == signal and signal != "HOLD":
                signal = "HOLD"

        signals_list.append(signal)
        conf_list.append(conf)

    return (
        pd.Series(signals_list, index=df_short.index),
        pd.Series(conf_list,    index=df_short.index),
    )


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def run_pipeline(
    ticker:    str,
    use_tft:   bool = False,
    use_gnn:   bool = False,
    use_macro: bool = True,
    all_dfs:   Optional[dict] = None,
) -> PipelineResult:
    """
    Run the full analysis pipeline for one ticker.

    Trains fresh every time — Streamlit's @st.cache_data in app.py
    ensures this only runs once per ticker per session (ttl=3600).

    Parameters
    ----------
    ticker    : NSE ticker e.g. 'TCS.NS'
    use_tft   : enable TFT model
    use_gnn   : enable GNN model
    use_macro : fetch India VIX and USD/INR
    all_dfs   : pre-fetched stock data dict for GNN
    """

    # ── 1. Price data ─────────────────────────────────────
    try:
        df_long  = _download(ticker, "10y")
        df_short = _download(ticker, "2y")
    except Exception as exc:
        return _empty(f"Price fetch failed: {exc}")

    if df_long.empty or df_short.empty:
        return _empty(f"No price data for {ticker}")

    # ── 2. All signal engines ─────────────────────────────
    from fundamentals        import get_fundamentals
    from fundamental_engine  import calculate_fundamental_score
    from sentiment           import get_news_sentiment
    from regime              import get_market_regime
    from nse_data            import get_all_nse_features
    from technical_engine    import calculate_technical_score

    fundamental_data = get_fundamentals(ticker)
    fund      = calculate_fundamental_score(fundamental_data)
    sentiment = get_news_sentiment(ticker)
    regime    = get_market_regime()
    nse       = get_all_nse_features(ticker)
    tech      = calculate_technical_score(df_short, fetch_macro=use_macro)

    # ── 3. ML model ───────────────────────────────────────
    from ml_model import prepare_features, predict_signal, train_model

    try:
        df_feat              = prepare_features(df_long)
        ml_result            = train_model(df_feat)
        df_feat_s            = prepare_features(df_short)
        ml_signal, ml_conf   = (
            predict_signal(ml_result, df_feat_s)
            if ml_result.is_valid and len(df_feat_s) >= SEQUENCE_LEN
            else ("NO_SIGNAL", 0.5)
        )
    except Exception as exc:
        logger.error("ML failed: %s", exc)
        from ml_model import MLResult, FEATURES
        ml_result  = MLResult(model=None, xgb_model=None, scaler=None,
                               features=FEATURES, metrics={})
        ml_signal, ml_conf = "NO_SIGNAL", 0.5

    ml_confidence = ml_conf

    # ── 4. TFT (optional) ────────────────────────────────
    tft_signal, tft_confidence, tft_result = "NO_SIGNAL", 0.5, None
    if use_tft:
        try:
            from tft_model import prepare_tft_data, predict_tft, train_tft
            tft_result = train_tft(prepare_tft_data(df_long, ticker_id=ticker))
            tft_signal, tft_confidence = predict_tft(
                tft_result, prepare_tft_data(df_short, ticker_id=ticker))
        except Exception as exc:
            logger.warning("TFT: %s", exc)

    # ── 5. GNN (optional) ────────────────────────────────
    gnn_signal, gnn_confidence, gnn_result = "NO_SIGNAL", 0.5, None
    if use_gnn and all_dfs:
        try:
            from gnn_model import predict_gnn, train_gnn
            gnn_result = train_gnn(all_dfs)
            gnn_signal, gnn_confidence = predict_gnn(gnn_result, all_dfs, ticker)
        except Exception as exc:
            logger.warning("GNN: %s", exc)

    # ── 6. Final decision ─────────────────────────────────
    from decision_engine import make_decision

    final = make_decision(
        tech, fund, sentiment,
        ml_signal, ml_confidence,
        tft_signal, tft_confidence,
        gnn_signal, gnn_confidence,
        market_regime=regime["regime"],
        nse_features=nse,
    )

    # ── 7. Backtest ───────────────────────────────────────
    from backtesting  import calculate_metrics, run_backtest
    from benchmark    import compare_to_benchmark, get_nifty_benchmark

    hist_regime         = _hist_regime(df_long, df_short)
    signals, confidences = _build_signals(
        df_long, df_short, hist_regime,
        ml_result, df_feat, fund, nse,
    )

    df_bt, trades = run_backtest(df_short, signals, confidences, INITIAL_CASH)
    metrics       = calculate_metrics(df_bt)
    benchmark     = get_nifty_benchmark(df_short.index[0], df_short.index[-1])
    comparison    = compare_to_benchmark(metrics, benchmark)

    return PipelineResult(
        final=final, tech=tech, fund=fund,
        sentiment=sentiment, regime=regime, nse=nse,
        ml_result=ml_result,
        ml_signal=ml_signal, ml_confidence=ml_confidence,
        tft_result=tft_result, tft_signal=tft_signal, tft_confidence=tft_confidence,
        gnn_result=gnn_result, gnn_signal=gnn_signal, gnn_confidence=gnn_confidence,
        fundamental_data=fundamental_data,
        df_short=df_short, df_bt=df_bt,
        signals=signals, confidences=confidences,
        hist_regime=hist_regime,
        trades=trades, metrics=metrics,
        benchmark=benchmark, comparison=comparison,
        cache_status="trained_live",
    )
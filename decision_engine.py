"""
decision_engine.py — Weighted final recommendation.

Changes:
  - Market regime filter: suppress BUY in BEAR market
  - Confidence gate: ignore ML signal below 60%
  - TFT + GNN signals wired into final score
  - Weights adjusted for 5-signal ensemble
"""

import logging

logger = logging.getLogger(__name__)

# Weights when all 5 signals available (must sum to 1.0)
_W_FULL = dict(
    technical   = 0.25,
    fundamental = 0.25,
    sentiment   = 0.10,
    ml          = 0.20,
    tft         = 0.10,
    gnn         = 0.10,
)

# Weights when only core 3 available (no TFT/GNN)
_W_CORE = dict(
    technical   = 0.30,
    fundamental = 0.30,
    sentiment   = 0.10,
    ml          = 0.30,
)

_BUY_THRESHOLD      = 0.52
_SELL_THRESHOLD     = 0.45
_WEAK_TECH          = 0.35
_STRONG_TECH        = 0.65
_MIN_ML_CONFIDENCE  = 0.60   # ignore ML signal below this


def _ml_to_score(signal: str, confidence: float = 0.5) -> float:
    if signal == "NO_SIGNAL":
        return 0.5
    if signal == "BUY":
        return float(confidence)
    return 1.0 - float(confidence)


def make_decision(
    tech:               dict,
    fund:               dict,
    sentiment:          dict,
    ml_signal:          str,
    ml_confidence:      float = 0.5,
    tft_signal:         str   = "NO_SIGNAL",
    tft_confidence:     float = 0.5,
    gnn_signal:         str   = "NO_SIGNAL",
    gnn_confidence:     float = 0.5,
    market_regime:      str   = "UNKNOWN",
    nse_features:       dict  = None,
) -> dict:
    """
    Combine up to 5 signals into a final weighted recommendation.

    Parameters
    ----------
    tech/fund/sentiment : engine result dicts
    ml_signal/confidence: LSTM prediction + sigmoid probability
    tft_signal/confidence: TFT prediction (optional)
    gnn_signal/confidence: GNN prediction (optional)
    market_regime       : 'BULL' | 'BEAR' | 'UNKNOWN' from regime.py

    Returns
    -------
    dict: signal, confidence, ml_confidence, final_score,
          component_scores, summary, risks, regime_override
    """
    tech_score = float(tech.get("score", 0.5))
    fund_score = float(fund.get("score", 0.5))
    sent_score = float(sentiment.get("normalised", (sentiment.get("score", 0) + 1) / 2))

    # ── Confidence gate ───────────────────────────────────
    effective_ml = ml_signal if ml_confidence >= _MIN_ML_CONFIDENCE else "NO_SIGNAL"
    ml_score     = _ml_to_score(effective_ml, ml_confidence)

    tft_active   = tft_signal not in ("NO_SIGNAL", "disabled")
    gnn_active   = gnn_signal not in ("NO_SIGNAL", "disabled")
    tft_score    = _ml_to_score(tft_signal, tft_confidence) if tft_active else None
    gnn_score    = _ml_to_score(gnn_signal, gnn_confidence) if gnn_active else None

    # ── Weighted score ────────────────────────────────────
    if tft_active and gnn_active:
        w = _W_FULL
        final_score = (
            tech_score  * w["technical"]   +
            fund_score  * w["fundamental"] +
            sent_score  * w["sentiment"]   +
            ml_score    * w["ml"]          +
            tft_score   * w["tft"]         +
            gnn_score   * w["gnn"]
        )
    elif tft_active or gnn_active:
        extra_score = tft_score if tft_active else gnn_score
        final_score = (
            tech_score  * 0.25 +
            fund_score  * 0.25 +
            sent_score  * 0.10 +
            ml_score    * 0.25 +
            extra_score * 0.15
        )
    else:
        w = _W_CORE
        final_score = (
            tech_score * w["technical"]   +
            fund_score * w["fundamental"] +
            sent_score * w["sentiment"]   +
            ml_score   * w["ml"]
        )

    # ── Base signal ───────────────────────────────────────
    if final_score > _BUY_THRESHOLD:
        signal = "BUY"
    elif final_score < _SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    # ── Sanity filters ────────────────────────────────────
    regime_override = None

    if signal == "BUY" and tech_score < _WEAK_TECH:
        signal = "HOLD"
        logger.debug("BUY→HOLD: weak tech %.2f", tech_score)

    elif signal == "SELL" and tech_score > _STRONG_TECH:
        signal = "HOLD"
        logger.debug("SELL→HOLD: strong tech %.2f", tech_score)

    # ── NSE features override ─────────────────────────────
    nse_info = nse_features or {}
    earnings_risk = nse_info.get("earnings_risk", "LOW")
    pcr_signal    = nse_info.get("pcr_signal", "NEUTRAL")
    sector_signal = nse_info.get("sector_signal", "UNKNOWN")

    # Block BUY within 2 weeks of earnings (too much event risk)
    if signal == "BUY" and earnings_risk == "HIGH":
        signal = "HOLD"
        logger.info("make_decision: BUY→HOLD — earnings within 14 days")

    # Boost confidence if sector is also strong
    if signal == "BUY" and sector_signal == "STRONG":
        final_score = min(1.0, final_score * 1.05)
        logger.debug("make_decision: sector boost applied")

    # PCR contrarian: extreme fear (PCR>1.5) = contrarian BUY signal
    if signal == "HOLD" and pcr_signal == "EXTREME FEAR" and tech_score > 0.45:
        signal = "BUY"
        logger.info("make_decision: HOLD→BUY — PCR extreme fear (contrarian)")

    # ── Market regime filter ──────────────────────────────
    if market_regime == "BEAR" and signal == "BUY":
        signal = "HOLD"
        regime_override = "BUY suppressed — Nifty50 in BEAR market (MA50 < MA200)"
        logger.info("Regime override: BUY → HOLD (BEAR market)")

    return {
        "signal":         signal,
        "confidence":     round(final_score * 100, 2),
        "ml_confidence":  round(ml_confidence * 100, 1),
        "final_score":    round(final_score, 4),
        "regime_override": regime_override,
        "earnings_risk":  earnings_risk,
        "pcr_signal":     pcr_signal,
        "sector_signal":  sector_signal,
        "component_scores": {
            "technical":   round(tech_score, 4),
            "fundamental": round(fund_score, 4),
            "sentiment":   round(sent_score, 4),
            "ml":          round(ml_score,   4),
            "tft":         round(tft_score,  4) if tft_score is not None else None,
            "gnn":         round(gnn_score,  4) if gnn_score is not None else None,
        },
        "summary": tech.get("buy_reasons", []) + fund.get("buy_reasons", []),
        "risks":   tech.get("sell_reasons", []) + fund.get("sell_reasons", []),
    }
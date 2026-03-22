# ==============================
# DECISION ENGINE
# ==============================

def make_decision(tech, fund, sentiment, ml_signal):

    # ==============================
    # EXTRACT SCORES
    # ==============================
    tech_score = tech["score"]
    fund_score = fund["score"]
    sent_score = sentiment["score"]

    # ==============================
    # ML SCORE (NORMALIZED)
    # ==============================
    if ml_signal == "BUY":
        ml_score = 1
    elif ml_signal == "SELL":
        ml_score = 0
    else:
        ml_score = 0.5

    # ==============================
    # FINAL WEIGHTED SCORE
    # ==============================
    final_score = (
        tech_score * 0.35 +
        fund_score * 0.35 +
        sent_score * 0.10 +
        ml_score * 0.20
    )

    # ==============================
    # BASE SIGNAL (BALANCED)
    # ==============================
    if final_score > 0.55:
        signal = "BUY"
    elif final_score < 0.45:
        signal = "SELL"
    else:
        signal = "HOLD"

    # ==============================
    # SMART FILTER (LIGHT, NOT STRICT)
    # ==============================
    # Avoid buying in weak trend
    if signal == "BUY" and tech_score < 0.4:
        signal = "HOLD"

    # Avoid selling in strong trend
    elif signal == "SELL" and tech_score > 0.6:
        signal = "HOLD"

    # ==============================
    # FINAL OUTPUT
    # ==============================
    return {
        "signal": signal,
        "confidence": round(final_score * 100, 2),
        "final_score": round(final_score, 2),
        "summary": tech["buy_reasons"] + fund["buy_reasons"],
        "risks": tech["sell_reasons"] + fund["sell_reasons"]
    }
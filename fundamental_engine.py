def calculate_fundamental_score(data):
    score = 0

    buy_reasons = []
    sell_reasons = []

    # ==============================
    # PROFITABILITY
    # ==============================
    if data["roe"] > 15:
        score += 1 * 0.15
        buy_reasons.append("High ROE (>15%) → strong profitability")
    elif data["roe"] < 10:
        score -= 1 * 0.15
        sell_reasons.append("Low ROE (<10%) → weak profitability")

    if data["operating_margin"] > 15:
        score += 1 * 0.10
        buy_reasons.append("Healthy operating margin")
    else:
        score -= 1 * 0.10
        sell_reasons.append("Low operating margin")

    # ==============================
    # VALUATION
    # ==============================
    if data["pe"] < data["industry_pe"]:
        score += 1 * 0.10
        buy_reasons.append("PE below industry → undervalued")
    else:
        score -= 1 * 0.10
        sell_reasons.append("PE above industry → overvalued")

    if data["peg"] < 1:
        score += 1 * 0.10
        buy_reasons.append("PEG < 1 → growth at reasonable price")
    elif data["peg"] > 1.5:
        score -= 1 * 0.10
        sell_reasons.append("PEG too high")

    if data["price_to_cash_flow"] < 30:
        score += 1 * 0.05
        buy_reasons.append("Good cash flow valuation")

    # ==============================
    # FINANCIAL HEALTH
    # ==============================
    if data["debt_to_equity"] < 1:
        score += 1 * 0.15
        buy_reasons.append("Low debt → financially stable")
    elif data["debt_to_equity"] > 2:
        score -= 1 * 0.15
        sell_reasons.append("High debt risk")

    # ==============================
    # GROWTH
    # ==============================
    if data["sales_growth"] > 10:
        score += 1 * 0.10
        buy_reasons.append("Strong sales growth")
    elif data["sales_growth"] < 0:
        score -= 1 * 0.10
        sell_reasons.append("Declining sales")

    if data["profit_growth"] > 10:
        score += 1 * 0.10
        buy_reasons.append("Strong profit growth")
    elif data["profit_growth"] < 0:
        score -= 1 * 0.10
        sell_reasons.append("Declining profits")

    # ==============================
    # OWNERSHIP
    # ==============================
    if data["promoter_holding"] > 50:
        score += 1 * 0.15
        buy_reasons.append("High promoter holding → strong confidence")
    elif data["promoter_holding"] < 30:
        score -= 1 * 0.15
        sell_reasons.append("Low promoter holding")

    # ==============================
    # FINAL SIGNAL
    # ==============================
    if score > 0.5:
        signal = "STRONG BUY"
    elif score > 0.2:
        signal = "BUY"
    elif score >= -0.2:
        signal = "HOLD"
    elif score >= -0.5:
        signal = "SELL"
    else:
        signal = "STRONG SELL"

    return {
        "score": round(score, 2),
        "signal": signal,
        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons
    }
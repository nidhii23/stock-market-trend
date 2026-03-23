"""
fundamental_engine.py — Score a stock based on fundamental data.

Fix: all scoring blocks now skip None values entirely instead of
     treating missing data as 0 (which was scoring them as bad/good incorrectly).
"""

import logging

logger = logging.getLogger(__name__)

_W = dict(
    roe              = 0.15,
    operating_margin = 0.10,
    pe               = 0.10,
    peg              = 0.10,
    price_to_cf      = 0.05,
    debt_to_equity   = 0.15,
    sales_growth     = 0.10,
    profit_growth    = 0.10,
    promoter_holding = 0.15,
)


def calculate_fundamental_score(data: dict) -> dict:
    """
    Score a stock 0–1 based on fundamental metrics.
    Fields with None values are skipped — weight is redistributed
    proportionally to available fields.
    """
    score         = 0.0
    weight_used   = 0.0
    buy_reasons   = []
    sell_reasons  = []

    def _v(key):
        """Return float value or None."""
        val = data.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    # ── Profitability ────────────────────────────────────
    roe = _v("roe")
    if roe is not None:
        weight_used += _W["roe"]
        if roe > 15:
            score += _W["roe"]
            buy_reasons.append(f"High ROE ({roe:.1f}%) → strong profitability")
        elif roe < 10:
            score -= _W["roe"]
            sell_reasons.append(f"Weak ROE ({roe:.1f}%) → poor returns")

    op_margin = _v("operating_margin")
    if op_margin is not None:
        weight_used += _W["operating_margin"]
        if op_margin > 15:
            score += _W["operating_margin"]
            buy_reasons.append(f"Healthy operating margin ({op_margin:.1f}%)")
        elif op_margin < 8:
            score -= _W["operating_margin"]
            sell_reasons.append(f"Thin operating margin ({op_margin:.1f}%)")

    # ── Valuation ────────────────────────────────────────
    pe, ind_pe = _v("pe"), _v("industry_pe")
    if pe is not None and ind_pe is not None and pe > 0 and ind_pe > 0:
        weight_used += _W["pe"]
        if pe < ind_pe:
            score += _W["pe"]
            buy_reasons.append(f"PE ({pe:.1f}x) below industry ({ind_pe:.1f}x) → cheap")
        else:
            score -= _W["pe"]
            sell_reasons.append(f"PE ({pe:.1f}x) above industry ({ind_pe:.1f}x) → pricey")

    peg = _v("peg")
    if peg is not None and peg > 0:          # skip if None OR zero (missing data)
        weight_used += _W["peg"]
        if peg < 1:
            score += _W["peg"]
            buy_reasons.append(f"PEG {peg:.2f} < 1 → growth at reasonable price")
        elif peg > 1.5:
            score -= _W["peg"]
            sell_reasons.append(f"PEG {peg:.2f} > 1.5 → expensive vs growth")

    pcf = _v("price_to_cash_flow")
    if pcf is not None and pcf > 0:
        weight_used += _W["price_to_cf"]
        if pcf < 30:
            score += _W["price_to_cf"]
            buy_reasons.append(f"Price/cash-flow {pcf:.1f}x → fair valuation")

    # ── Financial health ─────────────────────────────────
    d2e = _v("debt_to_equity")
    if d2e is not None:
        weight_used += _W["debt_to_equity"]
        if d2e < 1:
            score += _W["debt_to_equity"]
            buy_reasons.append(f"Debt/equity {d2e:.2f} → low leverage")
        elif d2e > 2:
            score -= _W["debt_to_equity"]
            sell_reasons.append(f"Debt/equity {d2e:.2f} → high financial risk")

    # ── Growth ───────────────────────────────────────────
    sg = _v("sales_growth")
    if sg is not None:
        weight_used += _W["sales_growth"]
        if sg > 10:
            score += _W["sales_growth"]
            buy_reasons.append(f"Sales growth {sg:.1f}% → strong expansion")
        elif sg < 0:
            score -= _W["sales_growth"]
            sell_reasons.append(f"Sales declining ({sg:.1f}%)")

    pg = _v("profit_growth")
    if pg is not None:
        weight_used += _W["profit_growth"]
        if pg > 10:
            score += _W["profit_growth"]
            buy_reasons.append(f"Profit growth {pg:.1f}% → strong earnings")
        elif pg < 0:
            score -= _W["profit_growth"]
            sell_reasons.append(f"Profits declining ({pg:.1f}%)")

    # ── Ownership ────────────────────────────────────────
    ph = _v("promoter_holding")
    if ph is not None:
        weight_used += _W["promoter_holding"]
        if ph > 50:
            score += _W["promoter_holding"]
            buy_reasons.append(f"Insider holding {ph:.1f}% → strong conviction")
        elif ph < 30:
            score -= _W["promoter_holding"]
            sell_reasons.append(f"Low insider holding ({ph:.1f}%)")

    # ── Normalise by weight actually used ────────────────
    if weight_used == 0:
        score_norm = 0.5   # no data → neutral
    else:
        # Scale score to [-1, 1] relative to weight used, then to [0, 1]
        score_scaled = score / weight_used   # [-1, 1]
        score_norm   = round((score_scaled + 1) / 2, 4)

    score_norm = max(0.0, min(1.0, score_norm))

    if score_norm > 0.75:   signal = "STRONG BUY"
    elif score_norm > 0.55: signal = "BUY"
    elif score_norm >= 0.45:signal = "HOLD"
    elif score_norm >= 0.25:signal = "SELL"
    else:                   signal = "STRONG SELL"

    return {
        "score":        score_norm,
        "signal":       signal,
        "buy_reasons":  buy_reasons,
        "sell_reasons": sell_reasons,
        "fields_scored": round(weight_used, 2),
    }
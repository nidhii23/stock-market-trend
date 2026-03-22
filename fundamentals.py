import yfinance as yf


def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    # Safe getter
    def safe_get(key, default=0):
        value = info.get(key, default)
        return value if value is not None else default

    fundamentals = {
        "roe": safe_get("returnOnEquity") * 100,
        "operating_margin": safe_get("operatingMargins") * 100,
        "pe": safe_get("trailingPE"),
        "industry_pe": safe_get("forwardPE"),
        "peg": safe_get("pegRatio"),
        "price_to_cash_flow": safe_get("priceToCashflow"),
        "debt_to_equity": safe_get("debtToEquity") / 100,
        "sales_growth": safe_get("revenueGrowth") * 100,
        "profit_growth": safe_get("earningsGrowth") * 100,
        "promoter_holding": safe_get("heldPercentInsiders") * 100
    }

    return fundamentals
import pandas as pd
import ta


def calculate_technical_score(df):
    df = df.copy()

    # ==============================
    # HANDLE MULTIINDEX (IMPORTANT)
    # ==============================
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ==============================
    # INDICATORS
    # ==============================
    df['ema_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()

    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()

    df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()

    # ==============================
    # LATEST VALUES
    # ==============================
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0

    buy_reasons = []
    sell_reasons = []

    # ==============================
    # EMA TREND
    # ==============================
    # Strong trend bonus
    if latest['Close'] > latest['ema_200']:
        score += 0.2   # NOT 0.3
        buy_reasons.append("Price above 200 EMA → uptrend")

    # ==============================
    # MACD
    # ==============================
    if latest['macd'] > latest['macd_signal']:
        score += 0.2
        buy_reasons.append("MACD bullish crossover")
    else:
        sell_reasons.append("MACD bearish crossover")

    # ==============================
    # RSI
    # ==============================
    if latest['rsi'] < 30:
        score += 0.15
        buy_reasons.append("RSI oversold → potential buy")
    elif latest['rsi'] > 70:
        sell_reasons.append("RSI overbought → potential sell")

    # ==============================
    # BOLLINGER BANDS
    # ==============================
    if latest['Close'] <= latest['bb_low']:
        score += 0.15
        buy_reasons.append("Price near lower Bollinger Band")
    elif latest['Close'] >= latest['bb_high']:
        sell_reasons.append("Price near upper Bollinger Band")

    # ==============================
    # VOLUME
    # ==============================
    if latest['Volume'] > prev['Volume'] and latest['Close'] > prev['Close']:
        score += 0.1
        buy_reasons.append("Volume increasing with price")
    elif latest['Volume'] > prev['Volume'] and latest['Close'] < prev['Close']:
        sell_reasons.append("Selling pressure with high volume")

    # ==============================
    # SUPPORT / RESISTANCE
    # ==============================
    support = df['Low'].rolling(20).min().iloc[-1]
    resistance = df['High'].rolling(20).max().iloc[-1]

    if latest['Close'] <= support * 1.02:
        score += 0.1
        buy_reasons.append("Near support level")
    elif latest['Close'] >= resistance * 0.98:
        sell_reasons.append("Near resistance level")

    # ==============================
    # ADX (TREND STRENGTH)
    # ==============================
    if latest['adx'] > 25:
        score += 0.1
        buy_reasons.append("Strong trend (ADX > 25)")

    # ==============================
    # NORMALIZE SCORE (VERY IMPORTANT)
    # ==============================
    score = min(score, 1)

    # ==============================
    # FINAL SIGNAL
    # ==============================
    if score > 0.7:
        signal = "STRONG BUY"
    elif score > 0.5:
        signal = "BUY"
    elif score > 0.3:
        signal = "HOLD"
    else:
        signal = "SELL"

    return {
        "score": round(score, 2),
        "signal": signal,
        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons
    }
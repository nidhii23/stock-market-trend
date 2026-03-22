import pandas as pd


# ==============================
# BACKTEST FUNCTION
# ==============================
def run_backtest(df, signals):

    initial_cash = 100000
    cash = initial_cash
    shares = 0
    position = 0
    entry_price = 0

    portfolio_values = []
    trades = []

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        signal = signals.iloc[i]

        # ==============================
        # PREVENT DUPLICATE TRADES
        # ==============================
        if signal == "BUY" and position == 1:
            signal = "HOLD"

        if signal == "SELL" and position == 0:
            signal = "HOLD"

        # ==============================
        # BUY LOGIC
        # ==============================
        if signal == "BUY" and position == 0:
            shares = cash / price
            cash = 0
            position = 1
            entry_price = price
            trades.append(("BUY", df.index[i], price))

        # ==============================
        # SELL LOGIC (SMART EXIT)
        # ==============================
        elif position == 1:

            stop_loss = price < entry_price * 0.97   # -3% loss
            target = price > entry_price * 1.05      # +5% profit

            if signal == "SELL" or stop_loss or target:
                cash = shares * price
                shares = 0
                position = 0
                trades.append(("SELL", df.index[i], price))

        # ==============================
        # PORTFOLIO VALUE
        # ==============================
        if position == 1:
            portfolio = shares * price
        else:
            portfolio = cash

        portfolio_values.append(portfolio)

    df["Portfolio"] = portfolio_values

    return df, trades


# ==============================
# PERFORMANCE METRICS
# ==============================
def calculate_metrics(df):

    returns = df["Portfolio"].pct_change().dropna()

    total_return = (df["Portfolio"].iloc[-1] / df["Portfolio"].iloc[0] - 1) * 100

    if returns.std() != 0:
        sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5)
    else:
        sharpe_ratio = 0

    cumulative = df["Portfolio"]
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min() * 100

    return {
        "Total Return (%)": round(total_return, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Max Drawdown (%)": round(max_drawdown, 2)
    }
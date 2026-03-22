import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score


def prepare_features(df):
    df = df.copy()

    
    # ✅ Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
     df.columns = df.columns.get_level_values(0)

# ✅ Then clean spaces
    df.columns = df.columns.str.strip()

    # Returns
    df["return"] = df["Close"].pct_change()

    # Moving averages
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma20"] = df["Close"].rolling(20).mean()

    # Volatility
    df["volatility"] = df["return"].rolling(10).std()

    # Momentum
    df["momentum"] = df["Close"] - df["Close"].shift(5)

    # Trend
    df["trend"] = (df["ma5"] > df["ma20"]).astype(int)

    # Target
    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    df = df.dropna()

    return df


# ==============================
# TRAIN MODEL + METRICS
# ==============================
def train_model(df):
    features = ["return", "ma5", "ma10", "ma20", "volatility", "momentum", "trend"]
    X = df[features]
    y = df["target"]
    if len(df) < 50:
        return None, None, {"accuracy": 0, "precision": 0}
    # Train-test split
    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    if len(X_test) == 0:
       return None, None, {"accuracy": 0, "precision": 0}

    model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    df_test = df.iloc[split:].copy()
    df_test["prediction"] = y_pred

    df_test["strategy_return"] = df_test["return"] * df_test["prediction"]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2)
    }

    return model, features, metrics


# ==============================
# PREDICT SIGNAL
# ==============================
def predict_signal(model, df, features):
    latest = df[features].iloc[-1:]

    pred = model.predict(latest)[0]

    if pred == 1:
        return "BUY"
    else:
        return "SELL"
"""
ml_model.py — LSTM + XGBoost ensemble classifier.

Short-term improvements:
  - 10 new features: 52-week high/low distance, ROC, Stochastic,
    Williams %R, OBV ratio, CMF (FII/DII proxy), ATR ratio,
    sector relative strength vs Nifty50
  - Total features: 14 → 24
  - Class imbalance fix (pos_weight)
  - Ensemble LSTM 60% + XGBoost 40%
  - Walk-forward CV (3 folds)
  - Confidence gate at 55%
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────

FEATURES = [
    "return", "ma5", "ma10", "ma20",
    "volatility", "momentum", "trend",
    "rsi", "return_lag1", "return_lag2",
    "sin_5", "cos_5", "sin_21", "cos_21",
    "dist_52w_high", "dist_52w_low",
    "roc_10", "roc_21",
    "stoch_k", "williams_r",
    "cmf", "obv_ratio",
    "atr_ratio",
    "rel_strength",
    "delivery_pct", "delivery_signal",
    "earnings_proximity",
]

SEQUENCE_LEN   = 40
HIDDEN_SIZE    = 128     # increased from 64 — more features need more capacity
NUM_LAYERS     = 2
DROPOUT        = 0.3
EPOCHS         = 50
BATCH_SIZE     = 32
LR             = 1e-3
MIN_ROWS       = 200
TRAIN_RATIO    = 0.8
WF_SPLITS      = 3
MIN_CONFIDENCE = 0.55

LSTM_WEIGHT = 0.60
XGB_WEIGHT  = 0.40

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

# Cache Nifty50 data to avoid repeated downloads
_nifty_cache: dict = {}


# ── LSTM ──────────────────────────────────────────────────

class _LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.bn      = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h      = out[:, -1, :]
        h      = self.bn(h)
        return self.fc(self.dropout(h)).squeeze(-1)


# ── Result container ──────────────────────────────────────

@dataclass
class MLResult:
    model:     Optional[object]
    xgb_model: Optional[object]
    scaler:    Optional[object]
    features:  list
    metrics:   dict
    df_test:   Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def is_valid(self):
        return self.model is not None or self.xgb_model is not None


def _empty():
    return MLResult(
        model=None, xgb_model=None, scaler=None, features=FEATURES,
        metrics={k: 0 for k in [
            "accuracy", "precision", "recall", "f1",
            "wf_accuracy", "wf_precision", "wf_recall", "wf_f1",
            "cumulative_strategy_return", "n_train", "n_test", "wf_folds",
        ]},
    )


# ── Feature engineering ───────────────────────────────────

def _rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _fetch_nifty(start, end) -> pd.Series:
    """Fetch Nifty50 returns, cached."""
    key = f"{start}_{end}"
    if key in _nifty_cache:
        return _nifty_cache[key]
    try:
        df = yf.download("^NSEI", start=start, end=end,
                         interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        ret = df["Close"].pct_change()
        _nifty_cache[key] = ret
        return ret
    except Exception:
        return pd.Series(dtype=float)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 24 features including new additions:
    52-week positioning, ROC, Stochastic, Williams %R,
    CMF (FII/DII proxy), OBV ratio, ATR ratio, relative strength.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.strip()

    if "Close" not in df.columns:
        raise ValueError(f"'Close' not found. Got: {list(df.columns)}")

    close  = df["Close"]
    high   = df["High"]   if "High"   in df.columns else close
    low    = df["Low"]    if "Low"    in df.columns else close
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=df.index)
    t      = np.arange(len(df))

    # ── Original features ────────────────────────────────
    df["return"]      = close.pct_change()
    df["ma5"]         = close.rolling(5).mean()
    df["ma10"]        = close.rolling(10).mean()
    df["ma20"]        = close.rolling(20).mean()
    df["volatility"]  = df["return"].rolling(10).std()
    df["momentum"]    = close - close.shift(5)
    df["trend"]       = (df["ma5"] > df["ma20"]).astype(int)
    df["rsi"]         = _rsi(close)
    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)

    # ── Fourier features ─────────────────────────────────
    df["sin_5"]  = np.sin(2 * np.pi * t / 5)
    df["cos_5"]  = np.cos(2 * np.pi * t / 5)
    df["sin_21"] = np.sin(2 * np.pi * t / 21)
    df["cos_21"] = np.cos(2 * np.pi * t / 21)

    # ── NEW: 52-week high/low distance ───────────────────
    w52 = 252
    rolling_high = close.rolling(w52, min_periods=60).max()
    rolling_low  = close.rolling(w52, min_periods=60).min()
    df["dist_52w_high"] = (close - rolling_high) / rolling_high   # negative = below 52w high
    df["dist_52w_low"]  = (close - rolling_low)  / rolling_low    # positive = above 52w low

    # ── NEW: Rate of change ──────────────────────────────
    df["roc_10"] = close.pct_change(10)
    df["roc_21"] = close.pct_change(21)

    # ── NEW: Stochastic oscillator %K ────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stoch_k"] = ((close - low14) / (high14 - low14).replace(0, np.nan)) * 100

    # ── NEW: Williams %R ─────────────────────────────────
    df["williams_r"] = ((high14 - close) / (high14 - low14).replace(0, np.nan)) * -100

    # ── NEW: Chaikin Money Flow (CMF) — FII/DII proxy ────
    # CMF = sum(((close-low)-(high-close))/(high-low) * volume) / sum(volume) over 20 days
    hl    = (high - low).replace(0, np.nan)
    mfv   = ((close - low) - (high - close)) / hl * volume
    df["cmf"] = mfv.rolling(20).sum() / volume.rolling(20).sum()

    # ── NEW: OBV ratio ───────────────────────────────────
    # OBV / 20-day MA of OBV — shows if volume trend is accelerating
    obv          = (np.sign(df["return"]) * volume).cumsum()
    obv_ma        = obv.rolling(20).mean()
    df["obv_ratio"] = obv / obv_ma.replace(0, np.nan)

    # ── NEW: ATR ratio ───────────────────────────────────
    # Current ATR / 20-day average ATR — detects volatility regime changes
    tr    = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr   = tr.rolling(14).mean()
    atr20 = atr.rolling(20).mean()
    df["atr_ratio"] = atr / atr20.replace(0, np.nan)

    # ── NEW: Relative strength vs Nifty50 ────────────────
    # Stock 21-day return / Nifty 21-day return — >1 = outperforming
    try:
        start_str = str(df.index[0].date())
        end_str   = str(df.index[-1].date())
        nifty_ret = _fetch_nifty(start_str, end_str)
        nifty_21  = (1 + nifty_ret).rolling(21).apply(
            lambda x: x.prod(), raw=True) - 1
        stock_21  = (1 + df["return"]).rolling(21).apply(
            lambda x: x.prod(), raw=True) - 1
        nifty_21  = nifty_21.reindex(df.index).fillna(method="ffill")
        df["rel_strength"] = stock_21 / nifty_21.replace(0, np.nan)
        df["rel_strength"] = df["rel_strength"].clip(-5, 5)   # cap outliers
    except Exception:
        df["rel_strength"] = 1.0   # neutral if Nifty fetch fails

    # ── Delivery % placeholder (populated per-ticker in app) ──
    if "delivery_pct" not in df.columns:
        df["delivery_pct"]    = 50.0
        df["delivery_signal"] = 0.0

    # ── Earnings proximity placeholder ───────────────────
    if "earnings_proximity" not in df.columns:
        df["earnings_proximity"] = 0.0

    # ── Target — improved: filter out noise ──────────────
    # Only label as UP if move exceeds 0.3x rolling daily volatility
    # This removes noise from tiny moves that don't matter
    daily_vol = df["return"].rolling(20).std()
    threshold = (daily_vol * 0.3).fillna(0)
    df["target"] = (df["return"].shift(-1) > threshold).astype(int)

    return df.dropna(subset=FEATURES + ["target"]).copy()


# ── Sequence builder ──────────────────────────────────────

def _sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ── Training helpers ──────────────────────────────────────

def _fit_lstm(X_tr, y_tr, X_vl, y_vl, input_size, pos_weight):
    model = _LSTMClassifier(input_size, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    Xtr, ytr = torch.tensor(X_tr), torch.tensor(y_tr)
    Xvl, yvl = torch.tensor(X_vl), torch.tensor(y_vl)
    best_loss, best_w = float("inf"), None

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xvl), yvl).item()
        sched.step(vl)
        if vl < best_loss:
            best_loss = vl
            best_w    = {k: v.clone() for k, v in model.state_dict().items()}

    if best_w:
        model.load_state_dict(best_w)
    return model


def _fit_xgb(X_tr, y_tr, X_vl, y_vl, scale_pos_weight):
    xgb = XGBClassifier(**{**XGB_PARAMS, "scale_pos_weight": scale_pos_weight})
    xgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
    return xgb


# ── Public API ────────────────────────────────────────────

def train_model(df: pd.DataFrame) -> MLResult:
    """
    Train LSTM+XGBoost ensemble with walk-forward CV.
    Returns MLResult; check .is_valid before use.
    """
    if len(df) < MIN_ROWS:
        logger.warning("train_model: %d rows < %d.", len(df), MIN_ROWS)
        return _empty()

    scaler = StandardScaler()
    X_all  = scaler.fit_transform(df[FEATURES].values).astype(np.float32)
    y_all  = df["target"].values.astype(np.float32)

    cw    = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_all)
    pos_w = float(cw[1] / cw[0])
    logger.info("pos_weight=%.3f", pos_w)

    # ── Walk-forward CV ───────────────────────────────────
    fold_size  = len(df) // (WF_SPLITS + 1)
    wf_results = []

    for fold in range(WF_SPLITS):
        tr_end = fold_size * (fold + 1)
        vl_end = tr_end + fold_size
        if vl_end > len(df):
            break

        Xtr_s, ytr_s = _sequences(X_all[:tr_end],       y_all[:tr_end],       SEQUENCE_LEN)
        Xvl_s, yvl_s = _sequences(X_all[tr_end:vl_end], y_all[tr_end:vl_end], SEQUENCE_LEN)

        if len(Xtr_s) < max(BATCH_SIZE, 100) or len(Xvl_s) < 20:
            continue

        m_lstm = _fit_lstm(Xtr_s, ytr_s, Xvl_s, yvl_s, len(FEATURES), pos_w)
        m_lstm.eval()
        with torch.no_grad():
            lstm_p = torch.sigmoid(m_lstm(torch.tensor(Xvl_s))).numpy()

        m_xgb  = _fit_xgb(Xtr_s[:, -1, :], ytr_s, Xvl_s[:, -1, :], yvl_s, pos_w)
        xgb_p  = m_xgb.predict_proba(Xvl_s[:, -1, :])[:, 1]

        yvl_t  = yvl_s[:len(lstm_p)]
        ens    = LSTM_WEIGHT * lstm_p + XGB_WEIGHT * xgb_p[:len(lstm_p)]
        preds  = (ens > 0.5).astype(int)

        wf_results.append({
            "accuracy":  accuracy_score(yvl_t, preds),
            "precision": precision_score(yvl_t, preds, zero_division=0),
            "recall":    recall_score(yvl_t, preds, zero_division=0),
            "f1":        f1_score(yvl_t, preds, zero_division=0),
        })

    # ── Final model on 80% ────────────────────────────────
    split           = int(len(df) * TRAIN_RATIO)
    Xtr_f, ytr_f    = _sequences(X_all[:split], y_all[:split], SEQUENCE_LEN)
    X_te_raw        = np.vstack([X_all[split - SEQUENCE_LEN:split], X_all[split:]])
    y_te_raw        = np.concatenate([y_all[split - SEQUENCE_LEN:split], y_all[split:]])
    Xte, yte        = _sequences(X_te_raw, y_te_raw, SEQUENCE_LEN)

    if len(Xtr_f) < BATCH_SIZE:
        return _empty()

    final_lstm = _fit_lstm(Xtr_f, ytr_f, Xte, yte, len(FEATURES), pos_w)
    final_xgb  = _fit_xgb(Xtr_f[:, -1, :], ytr_f, Xte[:, -1, :], yte[:len(Xte)], pos_w)

    final_lstm.eval()
    with torch.no_grad():
        lstm_p = torch.sigmoid(final_lstm(torch.tensor(Xte))).numpy()
    xgb_p  = final_xgb.predict_proba(Xte[:, -1, :])[:, 1]
    ens    = LSTM_WEIGHT * lstm_p + XGB_WEIGHT * xgb_p
    y_pred = (ens > 0.5).astype(int)
    yte_t  = yte[:len(y_pred)]

    df_test = df.iloc[split:split + len(y_pred)].copy()
    df_test["prediction"]      = y_pred
    df_test["confidence"]      = ens
    df_test["strategy_return"] = df_test["return"] * df_test["prediction"]

    def _avg(k):
        return round(float(np.mean([r[k] for r in wf_results])) * 100, 2) if wf_results else 0.0

    metrics = {
        "wf_accuracy":  _avg("accuracy"),
        "wf_precision": _avg("precision"),
        "wf_recall":    _avg("recall"),
        "wf_f1":        _avg("f1"),
        "accuracy":     round(accuracy_score(yte_t,  y_pred) * 100, 2),
        "precision":    round(precision_score(yte_t, y_pred, zero_division=0) * 100, 2),
        "recall":       round(recall_score(yte_t,    y_pred, zero_division=0) * 100, 2),
        "f1":           round(f1_score(yte_t,        y_pred, zero_division=0) * 100, 2),
        "cumulative_strategy_return": round(
            ((1 + df_test["strategy_return"]).prod() - 1) * 100, 2),
        "wf_folds":   len(wf_results),
        "n_train":    split,
        "n_test":     len(y_pred),
        "pos_weight": round(pos_w, 3),
        "n_features": len(FEATURES),
    }

    logger.info("Ensemble WF=%.1f%% holdout=%.1f%% features=%d",
                metrics["wf_accuracy"], metrics["accuracy"], len(FEATURES))

    return MLResult(model=final_lstm, xgb_model=final_xgb,
                    scaler=scaler, features=FEATURES,
                    metrics=metrics, df_test=df_test)


def predict_signal(result: MLResult, df: pd.DataFrame) -> tuple:
    """Return (signal, confidence) using LSTM+XGBoost ensemble."""
    if not result.is_valid:
        return "NO_SIGNAL", 0.5

    missing = [f for f in result.features if f not in df.columns]
    if missing:
        raise ValueError(f"predict_signal: missing features: {missing}")

    if len(df) < SEQUENCE_LEN:
        return "NO_SIGNAL", 0.5

    X_raw = result.scaler.transform(df[result.features].values[-SEQUENCE_LEN:])
    X_seq = torch.tensor(X_raw[np.newaxis].astype(np.float32))

    lstm_p, xgb_p = 0.5, 0.5

    if result.model is not None:
        result.model.eval()
        with torch.no_grad():
            lstm_p = float(torch.sigmoid(result.model(X_seq)).item())

    if result.xgb_model is not None:
        xgb_p = float(result.xgb_model.predict_proba(
            X_raw[-1:].astype(np.float32))[0, 1])

    prob_up    = LSTM_WEIGHT * lstm_p + XGB_WEIGHT * xgb_p
    signal     = "BUY" if prob_up >= 0.5 else "SELL"
    confidence = prob_up if signal == "BUY" else 1.0 - prob_up

    if confidence < MIN_CONFIDENCE:
        return "NO_SIGNAL", confidence

    return signal, confidence
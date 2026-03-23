"""
tft_model.py — Temporal Fusion Transformer for stock direction prediction.

The TFT (Lim et al., 2021) is the current best-practice architecture for
multi-horizon time-series forecasting. Key advantages over LSTM:
  - Variable selection network: learns WHICH features matter most
  - Gated residual networks: skips uninformative pathways
  - Multi-head attention: attends to relevant past time steps
  - Interpretable: outputs feature importance + attention weights

Uses pytorch-forecasting which wraps TFT with all the boilerplate.

Install:  pip install pytorch-forecasting pytorch-lightning
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

MIN_ROWS     = 200
TRAIN_RATIO  = 0.8
MAX_EPOCHS   = 20
BATCH_SIZE   = 64
MAX_ENCODER  = 60    # lookback window (days)
MAX_PREDICT  = 1     # predict 1 day ahead


@dataclass
class TFTResult:
    model:    Optional[object]
    metrics:  dict
    feature_importance: Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def is_valid(self):
        return self.model is not None


def _empty_tft():
    return TFTResult(model=None, metrics={"accuracy": 0, "mae": 0})


def prepare_tft_data(df: pd.DataFrame, ticker_id: str = "stock") -> pd.DataFrame:
    """
    Convert OHLCV df into the long format expected by pytorch-forecasting.
    Adds all technical features + Fourier features.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = df.columns.str.strip()

    close = df["Close"]
    df["return"]      = close.pct_change()
    df["ma5"]         = close.rolling(5).mean()
    df["ma10"]        = close.rolling(10).mean()
    df["ma20"]        = close.rolling(20).mean()
    df["volatility"]  = df["return"].rolling(10).std()
    df["momentum"]    = close - close.shift(5)
    df["trend"]       = (df["ma5"] > df["ma20"]).astype(int).astype(float)
    df["rsi"]         = _rsi(close)
    df["return_lag1"] = df["return"].shift(1)
    df["return_lag2"] = df["return"].shift(2)

    t = np.arange(len(df))
    df["sin_5"]  = np.sin(2 * np.pi * t / 5)
    df["cos_5"]  = np.cos(2 * np.pi * t / 5)
    df["sin_21"] = np.sin(2 * np.pi * t / 21)
    df["cos_21"] = np.cos(2 * np.pi * t / 21)

    # Target: next-day return (TFT predicts regression, we threshold to direction)
    df["target_return"] = df["return"].shift(-1)

    df = df.dropna().copy()
    df["time_idx"]  = np.arange(len(df))
    df["ticker"]    = ticker_id
    df["log_close"] = np.log(close[df.index])

    return df


def _rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def train_tft(df_tft: pd.DataFrame) -> TFTResult:
    """
    Train a Temporal Fusion Transformer on prepared TFT data.

    Parameters
    ----------
    df_tft : output of prepare_tft_data()

    Returns
    -------
    TFTResult with model, metrics, and feature importance DataFrame.
    """
    if len(df_tft) < MIN_ROWS:
        logger.warning("train_tft: only %d rows.", len(df_tft))
        return _empty_tft()

    try:
        import pytorch_lightning as pl
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.metrics import MAE, QuantileLoss
    except ImportError as e:
        logger.error("train_tft: pytorch-forecasting not installed. pip install pytorch-forecasting pytorch-lightning\n%s", e)
        return _empty_tft()

    time_features = [
        "return", "ma5", "ma10", "ma20", "volatility",
        "momentum", "rsi", "return_lag1", "return_lag2",
        "sin_5", "cos_5", "sin_21", "cos_21", "log_close",
    ]

    split      = int(len(df_tft) * TRAIN_RATIO)
    df_train   = df_tft.iloc[:split]
    df_val     = df_tft.iloc[split:]

    training = TimeSeriesDataSet(
        df_train,
        time_idx            = "time_idx",
        target              = "target_return",
        group_ids           = ["ticker"],
        max_encoder_length  = MAX_ENCODER,
        max_prediction_length = MAX_PREDICT,
        time_varying_known_reals   = ["time_idx"] + time_features,
        time_varying_unknown_reals = ["target_return"],
        add_relative_time_idx      = True,
        add_target_scales          = True,
        add_encoder_length         = True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df_val, predict=True, stop_randomization=True)

    train_dl = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
    val_dl   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate          = 1e-3,
        hidden_size            = 32,
        attention_head_size    = 2,
        dropout                = 0.2,
        hidden_continuous_size = 16,
        loss                   = QuantileLoss(),
        log_interval           = -1,
        reduce_on_plateau_patience = 3,
    )

    trainer = pl.Trainer(
        max_epochs          = MAX_EPOCHS,
        accelerator         = "auto",
        enable_progress_bar = False,
        enable_model_summary= False,
        logger              = False,
    )

    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # ── Evaluation ───────────────────────────────────────
    preds_raw = tft.predict(val_dl, return_y=True)
    y_pred_ret = preds_raw.output.median(dim=1).values.numpy() if hasattr(preds_raw.output, 'median') else preds_raw.output.numpy().mean(axis=1)
    y_true_ret = preds_raw.y[0].numpy()

    # Convert predicted return direction to accuracy
    y_pred_dir = (y_pred_ret > 0).astype(int)
    y_true_dir = (y_true_ret > 0).astype(int)
    accuracy   = float((y_pred_dir == y_true_dir).mean()) * 100
    mae        = float(np.abs(y_pred_ret - y_true_ret).mean())

    # ── Feature importance ───────────────────────────────
    try:
        interp     = tft.interpret_output(
            tft.predict(val_dl, mode="raw", return_x=True).output,
        )
        feat_imp   = pd.DataFrame({
            "feature":    time_features,
            "importance": interp["encoder_variables"].mean(0).numpy()[:len(time_features)],
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        feat_imp = None

    metrics = {
        "accuracy": round(accuracy, 2),
        "mae":      round(mae, 6),
        "n_train":  split,
        "n_val":    len(df_val),
    }

    logger.info("TFT — accuracy=%.1f%%  MAE=%.6f", accuracy, mae)
    return TFTResult(model=tft, metrics=metrics, feature_importance=feat_imp)


def predict_tft(result: TFTResult, df_tft: pd.DataFrame) -> tuple:
    """
    Return (signal, confidence) using the TFT model.

    signal     : 'BUY' | 'SELL' | 'NO_SIGNAL'
    confidence : 0.5–1.0
    """
    if not result.is_valid:
        return "NO_SIGNAL", 0.5

    try:
        from pytorch_forecasting import TimeSeriesDataSet
    except ImportError:
        return "NO_SIGNAL", 0.5

    try:
        # Use the last MAX_ENCODER rows for prediction
        df_pred = df_tft.iloc[-MAX_ENCODER:].copy()
        raw     = result.model.predict(
            df_pred, mode="raw", return_x=False
        )
        # Median prediction across quantiles
        pred_return = float(raw.output.median(dim=1).values.item())
        # Convert to directional signal with pseudo-confidence
        signal     = "BUY" if pred_return > 0 else "SELL"
        # Use absolute predicted return as proxy for confidence (clamp 0.5–0.95)
        confidence = min(0.95, max(0.5, 0.5 + abs(pred_return) * 10))
        return signal, confidence
    except Exception as exc:
        logger.warning("predict_tft failed: %s", exc)
        return "NO_SIGNAL", 0.5
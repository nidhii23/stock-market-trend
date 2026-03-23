"""
gnn_model.py — Graph Neural Network for inter-stock relationship modelling.

Research finding (2025): only 4.2% of stock prediction papers use relational
data, yet GNNs that model how shocks propagate across stocks consistently
outperform single-stock models — especially for correlated stocks like
Indian large-caps which share macro, sector, and FII flow exposures.

Architecture:
  - Build a correlation graph: edge weight = |correlation(returns_i, returns_j)|
  - Only keep edges above a threshold (prune noise)
  - Apply 2-layer Graph Convolutional Network (GCN)
  - Each node's hidden state combines its own LSTM features with
    messages from its neighbours — capturing contagion effects

Install:  pip install torch torch-geometric
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

CORR_THRESHOLD  = 0.3    # minimum |correlation| to keep an edge
GCN_HIDDEN      = 32
GCN_LAYERS      = 2
EPOCHS          = 20
LR              = 1e-3
SEQUENCE_LEN    = 20
MIN_STOCKS      = 2
MIN_ROWS        = 60


# ──────────────────────────────────────────────
# GCN definition
# ──────────────────────────────────────────────

def _build_gcn(in_channels, hidden, out_channels, layers):
    """Build a simple GCN using torch_geometric."""
    import torch.nn as nn
    from torch_geometric.nn import GCNConv

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(in_channels, hidden))
            for _ in range(layers - 2):
                self.convs.append(GCNConv(hidden, hidden))
            self.convs.append(GCNConv(hidden, out_channels))
            self.relu = nn.ReLU()

        def forward(self, x, edge_index, edge_weight=None):
            for i, conv in enumerate(self.convs[:-1]):
                x = self.relu(conv(x, edge_index, edge_weight))
            return self.convs[-1](x, edge_index, edge_weight)

    return GCN()


@dataclass
class GNNResult:
    model:       Optional[object]
    tickers:     list
    metrics:     dict
    edge_df:     Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def is_valid(self):
        return self.model is not None and len(self.tickers) >= MIN_STOCKS


def _empty_gnn():
    return GNNResult(model=None, tickers=[], metrics={"accuracy": 0})


# ──────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────

def build_correlation_graph(
    returns_dict: dict[str, pd.Series],
    threshold: float = CORR_THRESHOLD,
) -> tuple:
    """
    Build an adjacency graph from pairwise return correlations.

    Parameters
    ----------
    returns_dict : {ticker: pd.Series of daily returns}
    threshold    : minimum |correlation| to create an edge

    Returns
    -------
    edge_index  : (2, E) int tensor
    edge_weight : (E,) float tensor
    edge_df     : readable DataFrame of all edges
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch not installed. pip install torch")

    tickers = list(returns_dict.keys())
    n       = len(tickers)

    # Align all series on same index
    df_ret  = pd.DataFrame(returns_dict).dropna()
    corr    = df_ret.corr()

    src, dst, weights = [], [], []
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            c = abs(corr.iloc[i, j])
            if c >= threshold:
                src.append(i); dst.append(j); weights.append(c)
                src.append(j); dst.append(i); weights.append(c)
                edges.append({"from": tickers[i], "to": tickers[j],
                               "correlation": round(corr.iloc[i, j], 4)})

    if not src:
        logger.warning("build_correlation_graph: no edges above threshold %.2f", threshold)

    edge_index  = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    edge_df     = pd.DataFrame(edges)

    return edge_index, edge_weight, edge_df, tickers


# ──────────────────────────────────────────────
# Feature extraction per stock
# ──────────────────────────────────────────────

def extract_node_features(
    dfs: dict[str, pd.DataFrame],
    seq_len: int = SEQUENCE_LEN,
) -> tuple:
    """
    For each stock extract a feature vector (last seq_len rows averaged)
    to serve as node features in the GCN.

    Returns
    -------
    X : (n_stocks, n_features) numpy array
    y : (n_stocks,) binary targets (1=UP, 0=DOWN next day)
    """
    from ml_model import prepare_features

    Xs, ys = [], []
    for ticker, df in dfs.items():
        try:
            df_f = prepare_features(df)
            if len(df_f) < seq_len + 1:
                Xs.append(np.zeros(14))   # 14 = len(FEATURES) in ml_model
                ys.append(0)
                continue
            # Average over last seq_len rows as node feature vector
            feat_cols = [c for c in df_f.columns if c in [
                "return","ma5","ma10","ma20","volatility","momentum",
                "trend","rsi","return_lag1","return_lag2",
                "sin_5","cos_5","sin_21","cos_21"
            ]]
            x = df_f[feat_cols].iloc[-seq_len:].mean(axis=0).values
            y = int(df_f["target"].iloc[-1])
            Xs.append(x)
            ys.append(y)
        except Exception as exc:
            logger.warning("extract_node_features(%s): %s", ticker, exc)
            Xs.append(np.zeros(14))
            ys.append(0)

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_gnn(
    dfs: dict[str, pd.DataFrame],
    threshold: float = CORR_THRESHOLD,
) -> GNNResult:
    """
    Train a GCN on the stock correlation graph.

    Parameters
    ----------
    dfs       : {ticker: OHLCV DataFrame} for all stocks to model together
    threshold : minimum correlation to keep an edge

    Returns
    -------
    GNNResult
    """
    if len(dfs) < MIN_STOCKS:
        logger.warning("train_gnn: need at least %d stocks.", MIN_STOCKS)
        return _empty_gnn()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        logger.error("train_gnn: PyTorch not installed.")
        return _empty_gnn()

    # Build returns dict for correlation graph
    returns_dict = {}
    for ticker, df in dfs.items():
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy(); df.columns = df.columns.get_level_values(0)
        if "Close" in df.columns and len(df) >= MIN_ROWS:
            returns_dict[ticker] = df["Close"].pct_change().dropna()

    if len(returns_dict) < MIN_STOCKS:
        return _empty_gnn()

    edge_index, edge_weight, edge_df, tickers = build_correlation_graph(
        returns_dict, threshold
    )

    # Node features from all stocks (historical window)
    X_all, y_all = [], []
    # Use rolling windows for training samples
    min_len = min(len(v) for v in returns_dict.values())
    n_samples = min_len - SEQUENCE_LEN - 1

    if n_samples < 10:
        return _empty_gnn()

    # Build training data: for each time step, get node features + target
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    sample_Xs = []
    sample_ys = []

    for t in range(SEQUENCE_LEN, min_len - 1):
        row_x, row_y = [], []
        for ticker in tickers:
            df = dfs.get(ticker)
            if df is None or len(df) <= t:
                row_x.append(np.zeros(10))
                row_y.append(0)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy(); df.columns = df.columns.get_level_values(0)
            close   = df["Close"].values
            ret     = np.diff(close) / close[:-1]
            window  = ret[max(0, t-SEQUENCE_LEN):t]
            feat    = [
                float(np.mean(window)),
                float(np.std(window)) if len(window) > 1 else 0.0,
                float(window[-1]) if len(window) > 0 else 0.0,
                float(window[-2]) if len(window) > 1 else 0.0,
                float(np.max(window)) if len(window) > 0 else 0.0,
                float(np.min(window)) if len(window) > 0 else 0.0,
                float(np.mean(window[-5:])) if len(window) >= 5 else 0.0,
                float(np.mean(window[-10:])) if len(window) >= 10 else 0.0,
                float(close[t] - close[max(0, t-5)]) / (close[max(0, t-5)] + 1e-8),
                float(close[t] - close[max(0, t-20)]) / (close[max(0, t-20)] + 1e-8),
            ]
            target  = 1 if (t + 1 < len(ret) and ret[t] > 0) else 0
            row_x.append(feat)
            row_y.append(target)
        sample_Xs.append(row_x)
        sample_ys.append(row_y)

    n_feats = 10
    model   = _build_gcn(n_feats, GCN_HIDDEN, 1, GCN_LAYERS)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    crit    = nn.BCEWithLogitsLoss()

    split    = int(len(sample_Xs) * 0.8)
    all_preds, all_true = [], []

    for epoch in range(EPOCHS):
        model.train()
        for t in range(split):
            X_t = torch.tensor(sample_Xs[t], dtype=torch.float)
            y_t = torch.tensor(sample_ys[t], dtype=torch.float)
            opt.zero_grad()
            out  = model(X_t, edge_index, edge_weight).squeeze()
            loss = crit(out, y_t)
            loss.backward()
            opt.step()

    # Evaluate on holdout
    model.eval()
    with torch.no_grad():
        for t in range(split, len(sample_Xs)):
            X_t   = torch.tensor(sample_Xs[t], dtype=torch.float)
            y_t   = sample_ys[t]
            out   = torch.sigmoid(model(X_t, edge_index, edge_weight)).squeeze().numpy()
            preds = (out > 0.5).astype(int)
            all_preds.extend(preds.tolist())
            all_true.extend(y_t)

    accuracy = float(np.mean(np.array(all_preds) == np.array(all_true))) * 100 if all_preds else 0

    metrics = {
        "accuracy": round(accuracy, 2),
        "n_stocks": len(tickers),
        "n_edges":  len(edge_df) if edge_df is not None else 0,
    }

    logger.info("GNN — accuracy=%.1f%% on %d stocks, %d edges",
                accuracy, len(tickers), metrics["n_edges"])

    return GNNResult(model=model, tickers=tickers, metrics=metrics, edge_df=edge_df)


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

def predict_gnn(
    result: GNNResult,
    dfs: dict[str, pd.DataFrame],
    target_ticker: str,
) -> tuple:
    """
    Return (signal, confidence) for target_ticker using GNN context.

    Parameters
    ----------
    result        : output of train_gnn()
    dfs           : same {ticker: df} dict used for training
    target_ticker : which stock to get a signal for

    Returns
    -------
    signal     : 'BUY' | 'SELL' | 'NO_SIGNAL'
    confidence : 0.5–1.0
    """
    if not result.is_valid:
        return "NO_SIGNAL", 0.5

    if target_ticker not in result.tickers:
        logger.warning("predict_gnn: %s not in trained tickers.", target_ticker)
        return "NO_SIGNAL", 0.5

    try:
        import torch
        node_idx = result.tickers.index(target_ticker)

        row_x = []
        for ticker in result.tickers:
            df = dfs.get(ticker)
            if df is None:
                row_x.append([0.0] * 10)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy(); df.columns = df.columns.get_level_values(0)
            close  = df["Close"].values
            ret    = np.diff(close) / close[:-1]
            window = ret[-SEQUENCE_LEN:]
            feat   = [
                float(np.mean(window)),
                float(np.std(window)) if len(window) > 1 else 0.0,
                float(window[-1]) if len(window) > 0 else 0.0,
                float(window[-2]) if len(window) > 1 else 0.0,
                float(np.max(window)) if len(window) > 0 else 0.0,
                float(np.min(window)) if len(window) > 0 else 0.0,
                float(np.mean(window[-5:])) if len(window) >= 5 else 0.0,
                float(np.mean(window[-10:])) if len(window) >= 10 else 0.0,
                float(close[-1] - close[-6]) / (close[-6] + 1e-8),
                float(close[-1] - close[-21]) / (close[-21] + 1e-8),
            ]
            row_x.append(feat)

        X_t   = torch.tensor(row_x, dtype=torch.float)
        edge_index, edge_weight = _rebuild_edges(result)

        result.model.eval()
        with torch.no_grad():
            out = torch.sigmoid(
                result.model(X_t, edge_index, edge_weight)
            ).squeeze().numpy()

        prob_up    = float(out[node_idx]) if out.ndim > 0 else float(out)
        signal     = "BUY" if prob_up >= 0.5 else "SELL"
        confidence = prob_up if signal == "BUY" else 1.0 - prob_up
        return signal, confidence

    except Exception as exc:
        logger.warning("predict_gnn failed: %s", exc)
        return "NO_SIGNAL", 0.5


def _rebuild_edges(result: GNNResult):
    """Reconstruct edge tensors from the stored edge_df."""
    import torch
    if result.edge_df is None or result.edge_df.empty:
        n = len(result.tickers)
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)

    src, dst, w = [], [], []
    for _, row in result.edge_df.iterrows():
        i = result.tickers.index(row["from"])
        j = result.tickers.index(row["to"])
        c = abs(row["correlation"])
        src += [i, j]; dst += [j, i]; w += [c, c]

    return (torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(w, dtype=torch.float))
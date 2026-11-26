"""P2 (Tabular + Hybrid + SHAP) - shared loaders and evaluation helpers.

Inputs (all aligned to node order 0..203768 of processed_data.npz):
    data/processed/processed_data.npz           <- P0
    data/external/graph_features.parquet        <- P1
    outputs/predictions/gnn_embeddings.npy      <- P3
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "processed_data.npz"
GRAPH_FEAT_PATH = ROOT / "data" / "external" / "graph_features.parquet"
EMB_PATH = ROOT / "outputs" / "predictions" / "gnn_embeddings.npy"
PRED_DIR = ROOT / "outputs" / "predictions"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"

SEEDS = [42, 123, 456]
GRAPH_COLS = [
    "in_degree", "out_degree", "pagerank", "clustering", "core_number",
    "one_in_one_out_chain", "fan_out", "fan_in", "fan_out_fan_in_count",
    "in_directed_cycle", "triangle_count",
]
LOG_COLS = ["in_degree", "out_degree", "fan_out_fan_in_count", "triangle_count"]


def ensure_dirs():
    for d in (PRED_DIR, TABLE_DIR, FIG_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_base():
    d = np.load(DATA_PATH)
    return {k: d[k] for k in d.files}


def load_graph_features(tx_ids):
    """P1 file; verifies row order == P0 node order (no silent misalignment)."""
    g = pd.read_parquet(GRAPH_FEAT_PATH)
    assert len(g) == len(tx_ids), "graph_features row count != number of nodes"
    assert (g["tx_id"].to_numpy() == tx_ids).all(), "graph_features tx_id order differs from P0"
    out = g[GRAPH_COLS].copy()
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(np.int8)
    for c in LOG_COLS:                       # heavy-tailed counts
        out[c] = np.log1p(out[c])
    return out.to_numpy(np.float32), list(out.columns)


def load_embeddings(n_nodes):
    e = np.load(EMB_PATH)
    assert e.shape[0] == n_nodes, "embedding rows != number of nodes"
    return e.astype(np.float32)


def feature_sets(base, with_graph=True, with_emb=True):
    """name -> (matrix, column names). Timestep is never a feature."""
    loc = [f"local_{i+1}" for i in range(base["X_local"].shape[1])]
    agg = [f"agg_{i+1}" for i in range(base["X_aggregated"].shape[1])]
    sets = {
        "local93": (base["X_local"], loc),
        "full165": (base["X_full"], loc + agg),
    }
    if with_graph:
        G, gc = load_graph_features(base["tx_ids"])
        sets["local93+graph"] = (np.hstack([base["X_local"], G]), loc + gc)
        sets["full165+graph"] = (np.hstack([base["X_full"], G]), loc + agg + gc)
    if with_emb:
        E = load_embeddings(len(base["tx_ids"]))
        ec = [f"gnn_emb_{i}" for i in range(E.shape[1])]
        sets["local93+gnn"] = (np.hstack([base["X_local"], E]), loc + ec)
        sets["full165+gnn"] = (np.hstack([base["X_full"], E]), loc + agg + ec)
        if with_graph:
            sets["local93+graph+gnn"] = (np.hstack([base["X_local"], G, E]), loc + gc + ec)
    return sets


def split_idx(base):
    return (np.where(base["train_mask"])[0],
            np.where(base["val_mask"])[0],
            np.where(base["test_mask"])[0])


def eval_split(y, probs, idx, threshold=0.5):
    """Metrics for illicit class on labeled nodes only (unknown = -1 never passed in)."""
    yt, ps = y[idx], probs[idx]
    pred = ps >= threshold
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"pr_auc": float(average_precision_score(yt, ps)), "roc_auc": float(roc_auc_score(yt, ps)),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}

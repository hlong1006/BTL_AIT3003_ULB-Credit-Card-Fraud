# -*- coding: utf-8 -*-
"""
P2 (Quyen) - Tabular + Hybrid + SHAP
Shared utilities: load shared data (P0), graph features (P1), GNN embeddings (P3),
metrics compatible with P0's metrics.py conventions (PR-AUC = average precision).

Split (from P0 processed_data.npz):
  train_mask : timestep 1-28, labeled only   (25,207 nodes)
  val_mask   : timestep 29-34, labeled only  (4,687 nodes)
  test_mask  : timestep 35-49, labeled only  (16,670 nodes)
Unknown nodes (-1) are never used for loss/metrics.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]


def _first_existing(*candidates: Path) -> Path:
    """Support both layouts: team repo (data/, outputs/) and local workspace
    (downloads/data/, shared/outputs/)."""
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


DATA_NPZ = _first_existing(
    ROOT / "data" / "processed" / "processed_data.npz",
    ROOT / "downloads" / "data" / "processed" / "processed_data.npz",
)
SHARED_OUT = _first_existing(ROOT / "shared" / "outputs", ROOT / "outputs")
GRAPH_FEATURES_PARQUET = SHARED_OUT / "tables" / "graph_features.parquet"
GNN_EMBEDDINGS_NPY = SHARED_OUT / "predictions" / "gnn_embeddings.npy"
GNN_PROBS_NPY = SHARED_OUT / "predictions" / "best_gnn_probs.npy"

OUT = ROOT / "outputs" / "quyen"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PREDICTIONS = OUT / "predictions"
MODELS = OUT / "models"
for _d in (TABLES, FIGURES, PREDICTIONS, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
THRESHOLD = 0.5

GRAPH_FEATURE_COLS = [
    "in_degree", "out_degree", "pagerank", "clustering", "core_number",
    "one_in_one_out_chain", "fan_out", "fan_in", "fan_out_fan_in_count",
    "in_directed_cycle", "triangle_count",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_data() -> dict:
    """Load P0 processed data + P1 graph features, verify alignment."""
    d = np.load(DATA_NPZ)
    data = {k: d[k] for k in d.files}
    n = data["X_local"].shape[0]
    assert n == 203769, f"Expected 203769 nodes, got {n}"
    assert data["X_full"].shape == (n, 165)
    assert np.array_equal(data["X_full"][:, :93], data["X_local"]), \
        "X_full must be [X_local | X_aggregated]"
    # masks must be disjoint and labeled-only
    for m in ("train_mask", "val_mask", "test_mask"):
        assert np.all(data["y"][data[m]] >= 0), f"{m} contains unknown labels"
    assert not np.any(data["train_mask"] & data["val_mask"])
    assert not np.any(data["train_mask"] & data["test_mask"])
    assert not np.any(data["val_mask"] & data["test_mask"])

    gf = pd.read_parquet(GRAPH_FEATURES_PARQUET)
    gf = gf.sort_values("node_idx").reset_index(drop=True)
    assert np.array_equal(gf["node_idx"].values, np.arange(n)), \
        "graph_features.parquet must be ordered 0..N-1"
    assert np.array_equal(gf["tx_id"].values, data["tx_ids"]), \
        "graph_features tx_id order mismatch with processed data"
    data["X_graph"] = gf[GRAPH_FEATURE_COLS].to_numpy(dtype=np.float32)
    return data


def build_feature_configs(data: dict) -> dict:
    """Three tabular configurations required by the project spec.

    local93        : 93 local features (timestep EXCLUDED - anti-leakage)
    full165        : 93 local + 72 aggregated (1-hop) features
    local93+graph11: 93 local + 11 structural features from P1
    """
    local_names = [f"local_{i}" for i in range(1, 94)]
    agg_names = [f"agg_{i}" for i in range(1, 73)]
    graph_names = [f"graph_{c}" for c in GRAPH_FEATURE_COLS]
    return {
        "local93": {
            "X": data["X_local"],
            "names": local_names,
            "groups": ["local"] * 93,
        },
        "full165": {
            "X": data["X_full"],
            "names": local_names + agg_names,
            "groups": ["local"] * 93 + ["aggregated"] * 72,
        },
        "local93+graph11": {
            "X": np.hstack([data["X_local"], data["X_graph"]]).astype(np.float32),
            "names": local_names + graph_names,
            "groups": ["local"] * 93 + ["graph"] * 11,
        },
    }


def evaluate(y_true: np.ndarray, probs: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """Metrics on labeled nodes only. PR-AUC = average precision (P0 convention)."""
    y_true = np.asarray(y_true).astype(int)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n_samples": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "positive_rate": float((y_true == 1).mean()),
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def per_timestep_metrics(data: dict, probs: np.ndarray, mask_name: str = "test_mask",
                         threshold: float = THRESHOLD) -> pd.DataFrame:
    """F1/precision/recall/PR-AUC per timestep (for P4 drift analysis)."""
    mask = data[mask_name]
    ts = data["timestep"]
    y = data["y"]
    rows = []
    for t in np.unique(ts[mask]):
        m = mask & (ts == t)
        yt, pt = y[m], probs[m]
        row = {"timestep": int(t), "n_labeled": int(m.sum()),
               "n_illicit": int((yt == 1).sum())}
        if (yt == 1).sum() > 0 and (yt == 0).sum() > 0:
            row.update({k: v for k, v in evaluate(yt, pt, threshold).items()
                        if k in ("pr_auc", "precision", "recall", "f1", "tp", "fp", "fn")})
        else:
            row.update({"pr_auc": np.nan, "precision": np.nan, "recall": np.nan,
                        "f1": np.nan, "tp": np.nan, "fp": np.nan, "fn": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0

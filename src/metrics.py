
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score
)


# =========================================================
# 1. HELPER FUNCTIONS
# =========================================================

def _to_1d_array(array, name):
    """
    Convert input into a one-dimensional NumPy array.
    """

    array = np.asarray(array)

    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional. "
            f"Got shape: {array.shape}"
        )

    return array


def _prepare_labeled_data(y_true, y_score):
    """
    Validate inputs and remove unknown labels.

    Label convention:
        1  = Illicit
        0  = Licit
        -1 = Unknown

    Unknown nodes are excluded from evaluation.
    """

    y_true = _to_1d_array(y_true, "y_true")
    y_score = _to_1d_array(y_score, "y_score")

    if len(y_true) != len(y_score):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, "
            f"y_score={len(y_score)}"
        )

    # Validate labels
    valid_labels = np.isin(y_true, [-1, 0, 1])

    if not valid_labels.all():
        invalid = np.unique(y_true[~valid_labels])

        raise ValueError(
            f"Invalid labels detected: {invalid}"
        )

    # Remove unknown labels
    labeled_mask = y_true != -1

    y_true = y_true[labeled_mask].astype(np.int8)

    y_score = y_score[labeled_mask].astype(np.float64)

    if len(y_true) == 0:
        raise ValueError(
            "No labeled samples available for evaluation."
        )

    # Validate scores only on labeled samples
    if not np.isfinite(y_score).all():
        raise ValueError(
            "y_score contains NaN or Inf on labeled samples."
        )

    return y_true, y_score


# =========================================================
# 2. BINARY CLASSIFICATION METRICS
# =========================================================

def evaluate_binary_classifier(
    y_true,
    y_score,
    threshold=0.5
):
    """
    Evaluate binary classification for illicit detection.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels:
            1  = Illicit
            0  = Licit
           -1  = Unknown

    y_score : array-like
        Prediction scores.
        Higher scores indicate higher likelihood of illicit.

    threshold : float
        Threshold used to convert scores into predictions.

    Returns
    -------
    dict
        Classification metrics and confusion matrix.
    """

    # Validate threshold
    threshold = float(threshold)

    if not np.isfinite(threshold):
        raise ValueError(
            "Threshold must be finite."
        )

    # Remove unknown samples
    y_true, y_score = _prepare_labeled_data(
        y_true,
        y_score
    )

    # -----------------------------------------------------
    # 1. Convert scores to binary predictions
    # -----------------------------------------------------

    y_pred = (y_score >= threshold).astype(np.int8)

    # -----------------------------------------------------
    # 2. Confusion matrix
    # -----------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # -----------------------------------------------------
    # 3. Precision, Recall and F1
    # -----------------------------------------------------

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    # -----------------------------------------------------
    # 4. PR-AUC / Average Precision
    # -----------------------------------------------------

    positive_count = int((y_true == 1).sum())
    negative_count = int((y_true == 0).sum())

    if positive_count > 0:

        pr_auc = average_precision_score(
            y_true,
            y_score
        )

        # Trapezoidal PR-AUC for reference
        pr_precision, pr_recall, _ = (
            precision_recall_curve(
                y_true,
                y_score,
                pos_label=1
            )
        )

        pr_auc_trapezoid = auc(
            pr_recall,
            pr_precision
        )

    else:

        # PR-AUC is undefined when no positive samples exist
        pr_auc = np.nan
        pr_auc_trapezoid = np.nan

    # -----------------------------------------------------
    # 5. ROC-AUC (additional metric)
    # -----------------------------------------------------

    if positive_count > 0 and negative_count > 0:

        roc_auc = roc_auc_score(
            y_true,
            y_score
        )

    else:

        roc_auc = np.nan

    # -----------------------------------------------------
    # 6. Build result dictionary
    # -----------------------------------------------------

    metrics = {
        "n_samples": int(len(y_true)),

        "n_positive": positive_count,
        "n_negative": negative_count,

        "positive_rate": float(
            positive_count / len(y_true)
        ),

        "threshold": threshold,

        "pr_auc": float(pr_auc),
        "pr_auc_trapezoid": float(pr_auc_trapezoid),

        "roc_auc": float(roc_auc),

        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),

        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

    return metrics


# =========================================================
# 3. EVALUATION BY TIMESTEP
# =========================================================

def evaluate_by_timestep(
    y_true,
    y_score,
    timestep,
    threshold=0.5,
    mask=None
):
    """
    Evaluate classification performance separately
    for each timestep.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels for all nodes.

    y_score : array-like
        Prediction scores for all nodes.

    timestep : array-like
        Timestep associated with each node.

    threshold : float
        Classification threshold.

    mask : array-like or None
        Optional Boolean mask selecting timesteps/nodes
        to evaluate.

    Returns
    -------
    pandas.DataFrame
        One row per timestep.
    """

    y_true = _to_1d_array(y_true, "y_true")
    y_score = _to_1d_array(y_score, "y_score")
    timestep = _to_1d_array(timestep, "timestep")

    n = len(y_true)

    if len(y_score) != n or len(timestep) != n:
        raise ValueError(
            "y_true, y_score and timestep must have "
            "the same length."
        )

    # Default: evaluate every node
    if mask is None:

        mask = np.ones(
            n,
            dtype=bool
        )

    else:

        mask = _to_1d_array(mask, "mask")

        if len(mask) != n:
            raise ValueError(
                "Mask length does not match the dataset."
            )

        mask = mask.astype(bool)

    selected_timesteps = np.sort(
        np.unique(timestep[mask])
    )

    records = []

    for t in selected_timesteps:

        # Select nodes belonging to this timestep
        current_mask = mask & (timestep == t)

        current_y = y_true[current_mask]
        current_score = y_score[current_mask]

        # Exclude unknown labels
        labeled_count = np.sum(current_y != -1)

        if labeled_count == 0:

            result = {
                "n_samples": 0,
                "n_positive": 0,
                "n_negative": 0,
                "positive_rate": np.nan,
                "threshold": threshold,
                "pr_auc": np.nan,
                "pr_auc_trapezoid": np.nan,
                "roc_auc": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0
            }

        else:

            result = evaluate_binary_classifier(
                y_true=current_y,
                y_score=current_score,
                threshold=threshold
            )

        records.append({
            "timestep": int(t),
            **result
        })

    return pd.DataFrame(records)


# =========================================================
# 4. PRINT METRICS
# =========================================================

def print_metrics(metrics):
    """
    Print evaluation results in a consistent format.
    """

    print("\n" + "=" * 55)
    print("BINARY CLASSIFICATION EVALUATION")
    print("=" * 55)

    print(f"Number of samples : {metrics['n_samples']:,}")
    print(f"Illicit samples   : {metrics['n_positive']:,}")
    print(f"Licit samples     : {metrics['n_negative']:,}")

    print("\nRanking metrics:")

    print(f"PR-AUC (AP)       : {metrics['pr_auc']:.6f}")
    print(
        f"PR-AUC (trapezoid): "
        f"{metrics['pr_auc_trapezoid']:.6f}"
    )
    print(f"ROC-AUC           : {metrics['roc_auc']:.6f}")

    print("\nClassification metrics:")

    print(f"Threshold         : {metrics['threshold']:.4f}")
    print(f"Precision         : {metrics['precision']:.6f}")
    print(f"Recall            : {metrics['recall']:.6f}")
    print(f"F1-score          : {metrics['f1']:.6f}")

    print("\nConfusion matrix:")

    print(f"TP: {metrics['tp']:,}")
    print(f"FP: {metrics['fp']:,}")
    print(f"TN: {metrics['tn']:,}")
    print(f"FN: {metrics['fn']:,}")

    print("=" * 55)

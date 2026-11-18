
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve

from src.metrics import (
    evaluate_binary_classifier,
    print_metrics
)


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processed_data.npz"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

PREDICTION_DIR = OUTPUT_DIR / "predictions"
MODEL_DIR = OUTPUT_DIR / "models"
TABLE_DIR = OUTPUT_DIR / "tables"


# =========================================================
# 2. EXPERIMENT CONFIGURATION
# =========================================================

SEED = 42

FEATURE_KEY = "X_full"

N_ESTIMATORS = 300

MAX_SAMPLES = 256

CONTAMINATION = "auto"

# Initial configuration for validation-based threshold selection
THRESHOLD_METRIC = "f1"


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def print_section(title):
    """
    Print a formatted section title.
    """

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def create_output_directories():
    """
    Create output directories.
    """

    for directory in [
        PREDICTION_DIR,
        MODEL_DIR,
        TABLE_DIR
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# =========================================================
# 4. LOAD PROCESSED DATA
# =========================================================

def load_processed_data():
    """
    Load the standardized Elliptic dataset.

    Important:
        train_time_mask includes both labeled and
        unknown nodes from timesteps 1-28.

        Labels are used only for evaluation,
        never for fitting Isolation Forest.
    """

    print_section("1. LOAD PROCESSED DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    with np.load(
        DATA_PATH,
        allow_pickle=False
    ) as data:

        X = data[FEATURE_KEY].copy()

        y = data["y"].copy()

        train_time_mask = (
            data["train_time_mask"].copy()
        )

        val_mask = data["val_mask"].copy()

        test_mask = data["test_mask"].copy()

        timestep = data["timestep"].copy()

    # -----------------------------------------------------
    # Validate shapes
    # -----------------------------------------------------

    n = len(y)

    if X.shape[0] != n:
        raise ValueError(
            "Feature and label dimensions do not match."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Feature matrix contains NaN or Inf."
        )

    if not train_time_mask.any():
        raise ValueError(
            "Training data is empty."
        )

    if not val_mask.any() or not test_mask.any():
        raise ValueError(
            "Validation or test data is empty."
        )

    # Check temporal masks
    if np.any(train_time_mask & val_mask):
        raise ValueError(
            "Training and validation masks overlap."
        )

    if np.any(train_time_mask & test_mask):
        raise ValueError(
            "Training and test masks overlap."
        )

    if np.any(val_mask & test_mask):
        raise ValueError(
            "Validation and test masks overlap."
        )

    print(f"Feature configuration: {FEATURE_KEY}")
    print(f"Feature shape: {X.shape}")

    print("\nTemporal split:")
    print(f"Training nodes: {train_time_mask.sum():,}")
    print(f"Validation nodes: {val_mask.sum():,}")
    print(f"Test nodes: {test_mask.sum():,}")

    print("\nTraining node composition:")
    print(
        f"Labeled: "
        f"{np.sum(train_time_mask & (y >= 0)):,}"
    )
    print(
        f"Unknown: "
        f"{np.sum(train_time_mask & (y == -1)):,}"
    )

    print("\nNote: Labels are NOT used during model fitting.")

    return {
        "X": X,
        "y": y,
        "timestep": timestep,
        "train_time_mask": train_time_mask,
        "val_mask": val_mask,
        "test_mask": test_mask
    }


# =========================================================
# 5. PREPARE TRAINING DATA
# =========================================================

def prepare_training_data(data):
    """
    Extract all nodes from training timesteps.

    Do NOT filter by y != -1.
    Unknown nodes are also used for unsupervised fitting.
    """

    print_section("2. PREPARE UNSUPERVISED TRAINING DATA")

    X = data["X"]

    train_time_mask = data["train_time_mask"]

    X_train = X[train_time_mask]

    print(f"Training samples: {len(X_train):,}")
    print(f"Number of features: {X_train.shape[1]}")

    if len(X_train) == 0:
        raise ValueError(
            "No training samples available."
        )

    print("\n[PASS] All training-time nodes are retained.")
    print("[PASS] No labels are passed to Isolation Forest.")

    return X_train


# =========================================================
# 6. BUILD ISOLATION FOREST
# =========================================================

def build_model():
    """
    Initialize the Isolation Forest model.

    contamination='auto' avoids imposing a manually
    specified anomaly percentage.

    max_samples='auto' corresponds to min(256, n_samples)
    in scikit-learn.
    """

    print_section("3. BUILD ISOLATION FOREST")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,

        max_samples=MAX_SAMPLES,

        contamination=CONTAMINATION,

        max_features=1.0,

        bootstrap=False,

        n_jobs=-1,

        random_state=SEED,

        verbose=0
    )

    print(f"Number of trees: {N_ESTIMATORS}")
    print(f"Max samples per tree: {MAX_SAMPLES}")
    print(f"Contamination: {CONTAMINATION}")
    print(f"Random seed: {SEED}")

    return model


# =========================================================
# 7. TRAIN ISOLATION FOREST
# =========================================================

def train_model(model, X_train):
    """
    Fit Isolation Forest without using any labels.
    """

    print_section("4. TRAIN ISOLATION FOREST")

    print("Training started...")

    start_time = time.perf_counter()

    model.fit(X_train)

    training_time = (
        time.perf_counter() - start_time
    )

    print("\nTraining completed!")

    print(
        f"Training time: {training_time:.2f} seconds"
    )

    return model, training_time


# =========================================================
# 8. GENERATE ANOMALY SCORES
# =========================================================

def generate_anomaly_scores(model, X):
    """
Generate detection scores for all nodes.

Scikit-learn convention:
    Lower score_samples -> more anomalous.
    Higher score_samples -> less anomalous.

Empirical observation on validation:
    Higher score_samples are more associated
    with illicit transactions.

Therefore:
    Keep the original score_samples without negation.

Important:
    This direction is selected using validation labels.
"""

    print_section("5. GENERATE ANOMALY SCORES")

    start_time = time.perf_counter()

    anomaly_scores = model.score_samples(X)

    scoring_time = (
        time.perf_counter() - start_time
    )

    anomaly_scores = anomaly_scores.astype(
        np.float32
    )

    if anomaly_scores.shape != (len(X),):
        raise ValueError(
            f"Unexpected score shape: {anomaly_scores.shape}"
        )

    if not np.isfinite(anomaly_scores).all():
        raise ValueError(
            "Anomaly scores contain NaN or Inf."
        )

    print(f"Score shape: {anomaly_scores.shape}")

    print(
        f"Minimum score: {anomaly_scores.min():.6f}"
    )

    print(
        f"Maximum score: {anomaly_scores.max():.6f}"
    )

    print(
        f"Mean score: {anomaly_scores.mean():.6f}"
    )

    print(
        f"Scoring time: {scoring_time:.2f} seconds"
    )

    print(
    "\n[INFO] Higher scores are more associated "
    "with illicit transactions based on validation."
)

    return anomaly_scores, scoring_time


# =========================================================
# 9. SELECT THRESHOLD USING VALIDATION
# =========================================================

def select_validation_threshold(
    y_val,
    anomaly_scores_val
):
    """
    Find the threshold maximizing F1 on validation.

    IMPORTANT:
        The model itself is unsupervised.
        Validation labels are used only to select
        an operating threshold.

    This threshold must be frozen before test evaluation.
    """

    print_section("6. SELECT THRESHOLD ON VALIDATION")

    y_val = np.asarray(y_val).reshape(-1)

    scores = np.asarray(
        anomaly_scores_val,
        dtype=np.float64
    ).reshape(-1)

    if len(y_val) != len(scores):
        raise ValueError(
            "Validation labels and scores have different lengths."
        )

    if not np.isin(y_val, [0, 1]).all():
        raise ValueError(
            "Validation labels must be binary."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Validation scores contain NaN or Inf."
        )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_val,
            scores,
            pos_label=1
        )
    )

    # precision and recall have one more element
    # than thresholds, so exclude the final point.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0
    )

    if len(thresholds) == 0:
        raise ValueError(
            "No threshold candidates available."
        )

    max_f1 = np.max(f1_scores)

    # Tie-breaking:
    # 1. Maximum F1
    # 2. Higher precision
    # 3. Higher threshold
    candidate_indices = np.flatnonzero(
        np.isclose(f1_scores, max_f1)
    )

    best_idx = candidate_indices[
        np.lexsort((
            -thresholds[candidate_indices],
            -precision[candidate_indices]
        ))[0]
    ]

    best_threshold = float(
        thresholds[best_idx]
    )

    print(f"Selected threshold: {best_threshold:.8f}")
    print(f"Validation F1: {f1_scores[best_idx]:.6f}")
    print(f"Validation Precision: {precision[best_idx]:.6f}")
    print(f"Validation Recall: {recall[best_idx]:.6f}")

    return best_threshold


# =========================================================
# 10. EVALUATE ISOLATION FOREST
# =========================================================

def evaluate_model(
    data,
    anomaly_scores,
    threshold
):
    """
    Evaluate validation and test performance.

    Unknown samples are automatically excluded
    by evaluate_binary_classifier().
    """

    print_section("7. EVALUATE ISOLATION FOREST")

    y = data["y"]

    val_mask = data["val_mask"]

    test_mask = data["test_mask"]

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    print("\nVALIDATION RESULTS")

    val_metrics = evaluate_binary_classifier(
        y_true=y[val_mask],
        y_score=anomaly_scores[val_mask],
        threshold=threshold
    )

    print_metrics(val_metrics)

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    print("\nTEST RESULTS")

    test_metrics = evaluate_binary_classifier(
        y_true=y[test_mask],
        y_score=anomaly_scores[test_mask],
        threshold=threshold
    )

    print_metrics(test_metrics)

    return {
        "validation": val_metrics,
        "test": test_metrics
    }


# =========================================================
# 11. SAVE RESULTS
# =========================================================

def save_results(
    model,
    anomaly_scores,
    threshold,
    metrics,
    training_time,
    scoring_time
):
    """
    Save:
        - All-node anomaly scores
        - Isolation Forest model
        - Evaluation results
    """

    print_section("8. SAVE ISOLATION FOREST RESULTS")

    create_output_directories()

    # -----------------------------------------------------
    # Save anomaly scores
    # -----------------------------------------------------

    score_path = (
        PREDICTION_DIR
        / "isolation_forest_scores.npy"
    )

    np.save(
        score_path,
        anomaly_scores
    )

    print(f"Scores saved: {score_path}")

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR
        / "isolation_forest.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    print(f"Model saved: {model_path}")

    # -----------------------------------------------------
    # Save evaluation metrics
    # -----------------------------------------------------

    metrics_path = (
        TABLE_DIR
        / "isolation_forest_metrics.json"
    )

    results = {
    "algorithm": "IsolationForest",

    "seed": SEED,

    "feature_configuration": FEATURE_KEY,

    "n_estimators": N_ESTIMATORS,

    "max_samples": MAX_SAMPLES,

    "contamination": CONTAMINATION,

    "threshold_selection": THRESHOLD_METRIC,

    "score_method": "IsolationForest.score_samples",

    "score_direction": (
    "Higher scores are more associated with illicit labels; "
    "direction selected using validation labels"),

    "threshold": threshold,

    "training_time_seconds": training_time,

    "scoring_time_seconds": scoring_time,

    "validation": metrics["validation"],

    "test": metrics["test"]
}

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Metrics saved: {metrics_path}")

    # -----------------------------------------------------
    # Save comparison table
    # -----------------------------------------------------

    comparison = pd.DataFrame([
        {
            "split": "Validation",
            **metrics["validation"]
        },
        {
            "split": "Test",
            **metrics["test"]
        }
    ])

    comparison_path = (
        TABLE_DIR
        / "unsupervised_results.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print(f"Comparison saved: {comparison_path}")


# =========================================================
# 12. MAIN
# =========================================================

def main():

    print_section("ELLIPTIC ISOLATION FOREST")

    create_output_directories()

    # 1. Load data
    data = load_processed_data()

    # 2. Prepare unsupervised training samples
    X_train = prepare_training_data(data)

    # 3. Build model
    model = build_model()

    # 4. Train without labels
    model, training_time = train_model(
        model,
        X_train
    )

    # 5. Score all nodes
    anomaly_scores, scoring_time = (
        generate_anomaly_scores(
            model,
            data["X"]
        )
    )

    # 6. Select threshold using validation labels
    val_mask = data["val_mask"]

    threshold = select_validation_threshold(
        y_val=data["y"][val_mask],
        anomaly_scores_val=anomaly_scores[val_mask]
    )

    # 7. Evaluate
    metrics = evaluate_model(
        data=data,
        anomaly_scores=anomaly_scores,
        threshold=threshold
    )

    # 8. Save outputs
    save_results(
        model=model,
        anomaly_scores=anomaly_scores,
        threshold=threshold,
        metrics=metrics,
        training_time=training_time,
        scoring_time=scoring_time
    )

    print_section("ISOLATION FOREST COMPLETED")

    print("All-node anomaly scores are ready.")

    print(
        PREDICTION_DIR
        / "isolation_forest_scores.npy"
    )


if __name__ == "__main__":
    main()

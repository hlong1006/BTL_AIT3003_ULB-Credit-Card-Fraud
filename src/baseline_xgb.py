
import json
from inspect import signature
from pathlib import Path
import time
import numpy as np
import xgboost as xgb

from xgboost import XGBClassifier

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

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "models"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)


# =========================================================
# 2. EXPERIMENT CONFIGURATION
# =========================================================

SEED = 42

# Use all original features:
# 93 local + 72 aggregated
FEATURE_KEY = "X_full"

INITIAL_THRESHOLD = 0.5

EARLY_STOPPING_ROUNDS = 50


# =========================================================
# 3. XGBOOST HYPERPARAMETERS
# =========================================================

XGB_PARAMS = {
    "n_estimators": 2000,

    "max_depth": 6,
    "learning_rate": 0.03,

    "min_child_weight": 1,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "gamma": 0,

    "reg_alpha": 0,
    "reg_lambda": 1,

    "objective": "binary:logistic",
    "eval_metric": "aucpr",

    "tree_method": "hist",

    "random_state": SEED,
    "n_jobs": -1,

    "verbosity": 1
}


# =========================================================
# 4. HELPER FUNCTIONS
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
    Create directories for predictions, models and metrics.
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
# 5. LOAD PROCESSED DATA
# =========================================================

def load_processed_data():
    """
    Load the standardized dataset created by src.data.
    """

    print_section("1. LOAD PROCESSED DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    with np.load(
        DATA_PATH,
        allow_pickle=False
    ) as data:

        X = data[FEATURE_KEY].astype(
            np.float32,
            copy=False
        )

        y = data["y"].copy()

        timestep = data["timestep"].copy()

        tx_ids = data["tx_ids"].copy()

        train_mask = data["train_mask"].copy()
        val_mask = data["val_mask"].copy()
        test_mask = data["test_mask"].copy()

    print(f"Feature key: {FEATURE_KEY}")
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")

    print("\nTemporal split:")

    print(f"Train: {train_mask.sum():,}")
    print(f"Val  : {val_mask.sum():,}")
    print(f"Test : {test_mask.sum():,}")

    return {
        "X": X,
        "y": y,
        "timestep": timestep,
        "tx_ids": tx_ids,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask
    }


# =========================================================
# 6. PREPARE TRAIN / VALIDATION DATA
# =========================================================

def prepare_training_data(data):
    """
    Extract labeled train and validation samples.

    Unknown nodes are excluded from supervised training
    and validation.
    """

    print_section("2. PREPARE TRAINING DATA")

    X = data["X"]
    y = data["y"]

    train_mask = data["train_mask"]
    val_mask = data["val_mask"]

    # Train data
    X_train = X[train_mask]

    y_train = y[train_mask].astype(
        np.int32
    )

    # Validation data
    X_val = X[val_mask]

    y_val = y[val_mask].astype(
        np.int32
    )

    # Validate labels
    if not np.isin(y_train, [0, 1]).all():
        raise ValueError(
            "Train data contains invalid labels."
        )

    if not np.isin(y_val, [0, 1]).all():
        raise ValueError(
            "Validation data contains invalid labels."
        )

    # Check that both classes exist
    train_positive = int((y_train == 1).sum())
    train_negative = int((y_train == 0).sum())

    val_positive = int((y_val == 1).sum())
    val_negative = int((y_val == 0).sum())

    if train_positive == 0 or train_negative == 0:
        raise ValueError(
            "Train data must contain both classes."
        )

    if val_positive == 0 or val_negative == 0:
        raise ValueError(
            "Validation data must contain both classes."
        )

    print("\nTrain:")
    print(f"Samples : {len(y_train):,}")
    print(f"Illicit : {train_positive:,}")
    print(f"Licit   : {train_negative:,}")

    print("\nValidation:")
    print(f"Samples : {len(y_val):,}")
    print(f"Illicit : {val_positive:,}")
    print(f"Licit   : {val_negative:,}")

    return X_train, y_train, X_val, y_val


# =========================================================
# 7. CALCULATE CLASS IMBALANCE WEIGHT
# =========================================================

def calculate_scale_pos_weight(y_train):
    """
    Calculate:

        scale_pos_weight = N_negative / N_positive

    Only training labels are used.
    """

    print_section("3. HANDLE CLASS IMBALANCE")

    positive_count = int(
        (y_train == 1).sum()
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    if positive_count == 0:
        raise ValueError(
            "No positive samples in training data."
        )

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(f"Negative samples: {negative_count:,}")
    print(f"Positive samples: {positive_count:,}")

    print(
        "\nscale_pos_weight = "
        f"{negative_count} / {positive_count}"
    )

    print(
        f"scale_pos_weight = {scale_pos_weight:.4f}"
    )

    return scale_pos_weight


# =========================================================
# 8. BUILD XGBOOST MODEL
# =========================================================

def build_model(scale_pos_weight):
    """
    Initialize XGBoost binary classifier.

    Handles early_stopping_rounds compatibility
    across different XGBoost sklearn API versions.
    """

    params = XGB_PARAMS.copy()

    params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**params)

    # Recent XGBoost versions place
    # early_stopping_rounds in the constructor.
    #
    # Older sklearn wrappers expose it in fit().

    fit_parameters = signature(
        model.fit
    ).parameters

    if "early_stopping_rounds" in fit_parameters:

        early_stopping_in_fit = True

    else:

        early_stopping_in_fit = False

        model.set_params(
            early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )

    return model, early_stopping_in_fit


# =========================================================
# 9. TRAIN XGBOOST
# =========================================================


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    scale_pos_weight
):
    """
    Train XGBoost using train data.

    Validation data is only used for monitoring
    and early stopping.

    Returns:
        model
        training_time
    """

    print_section("4. TRAIN XGBOOST")

    model, early_stopping_in_fit = build_model(
        scale_pos_weight
    )

    fit_kwargs = {
        "eval_set": [
            (X_val, y_val)
        ],
        "verbose": 100
    }

    if early_stopping_in_fit:
        fit_kwargs["early_stopping_rounds"] = (
            EARLY_STOPPING_ROUNDS
        )

    print("Model configuration:")
    print(f"Number of trees: {XGB_PARAMS['n_estimators']}")
    print(f"Max depth: {XGB_PARAMS['max_depth']}")
    print(f"Learning rate: {XGB_PARAMS['learning_rate']}")
    print(f"Early stopping: {EARLY_STOPPING_ROUNDS}")
    print(f"Random seed: {SEED}")

    print("\nTraining started...")

    # Measure training time
    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        **fit_kwargs
    )

    training_time = (
        time.perf_counter() - start_time
    )

    print("\nTraining completed!")

    print(f"Training time: {training_time:.2f} seconds")

    if hasattr(model, "best_iteration"):

        print(
            f"Best iteration: {model.best_iteration}"
        )

        print(
            f"Best validation score: {model.best_score}"
        )

    return model, training_time



# =========================================================
# 10. GENERATE PREDICTIONS FOR ALL NODES
# =========================================================


def generate_predictions(model, X):
    """
    Generate illicit probabilities for ALL nodes.

    Output shape:
        (203769,)

    Measure scoring time over the entire feature matrix.
    """

    print_section("5. GENERATE PREDICTIONS")

    # Measure inference/scoring time
    start_time = time.perf_counter()

    probabilities = model.predict_proba(X)[:, 1]

    scoring_time = (
        time.perf_counter() - start_time
    )

    probabilities = probabilities.astype(
        np.float32
    )

    if probabilities.shape != (len(X),):
        raise ValueError(
            f"Unexpected probability shape: {probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Predictions contain NaN or Inf."
        )

    if np.any(
        (probabilities < 0)
        | (probabilities > 1)
    ):
        raise ValueError(
            "Predicted probabilities must be in [0, 1]."
        )

    print(f"Prediction shape: {probabilities.shape}")
    print(f"Minimum score: {probabilities.min():.6f}")
    print(f"Maximum score: {probabilities.max():.6f}")
    print(f"Mean score: {probabilities.mean():.6f}")

    print(f"Scoring time: {scoring_time:.4f} seconds")

    return probabilities, scoring_time



# =========================================================
# 11. EVALUATE MODEL
# =========================================================

def evaluate_model(model, data, probabilities):
    """
    Evaluate validation and test sets.

    Threshold = 0.5 is only the initial baseline.
    It will be optimized later in cost.py.
    """

    print_section("6. EVALUATE BASELINE")

    y = data["y"]

    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    # -----------------------------------------------------
    # Validation evaluation
    # -----------------------------------------------------

    val_metrics = evaluate_binary_classifier(
        y_true=y[val_mask],
        y_score=probabilities[val_mask],
        threshold=INITIAL_THRESHOLD
    )

    print("\nVALIDATION RESULTS")

    print_metrics(val_metrics)

    # -----------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------

    # Test is evaluated for reporting only.
    # It is NOT used for hyperparameter tuning.

    test_metrics = evaluate_binary_classifier(
        y_true=y[test_mask],
        y_score=probabilities[test_mask],
        threshold=INITIAL_THRESHOLD
    )

    print("\nTEST RESULTS")

    print_metrics(test_metrics)

    return {
        "validation": val_metrics,
        "test": test_metrics
    }


# =========================================================
# 12. SAVE MODEL AND RESULTS
# =========================================================


def save_results(
    model,
    probabilities,
    metrics,
    training_time,
    scoring_time
):
    """
    Save:
        - All-node prediction probabilities
        - Trained XGBoost model
        - Evaluation metrics
        - Runtime measurements
    """

    print_section("7. SAVE RESULTS")

    create_output_directories()

    # -----------------------------------------------------
    # Save all-node probabilities
    # -----------------------------------------------------

    prediction_path = (
        PREDICTION_DIR
        / "baseline_xgb_probs.npy"
    )

    np.save(
        prediction_path,
        probabilities
    )

    print(f"Predictions saved: {prediction_path}")

    # -----------------------------------------------------
    # Save trained model
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR
        / "baseline_xgb.json"
    )

    model.save_model(model_path)

    print(f"Model saved: {model_path}")

    # -----------------------------------------------------
    # Save metrics and runtime
    # -----------------------------------------------------

    metrics_path = (
        TABLE_DIR
        / "baseline_xgb_metrics.json"
    )

    results = {
        "algorithm": "XGBoost",

        "seed": SEED,

        "feature_configuration": FEATURE_KEY,

        "threshold": INITIAL_THRESHOLD,

        "training_time_seconds": float(training_time),

        "scoring_time_seconds": float(scoring_time),

        "total_time_seconds": float(
            training_time + scoring_time
        ),

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

    print("\nRuntime:")
    print(f"Training: {training_time:.2f} seconds")
    print(f"Scoring : {scoring_time:.4f} seconds")
    print(f"Total   : {training_time + scoring_time:.2f} seconds")

    print("\nFile sizes:")

    print(
        f"Predictions: "
        f"{prediction_path.stat().st_size / (1024**2):.2f} MB"
    )

    print(
        f"Model: "
        f"{model_path.stat().st_size / (1024**2):.2f} MB"
    )



# =========================================================
# 13. MAIN
# =========================================================

def main():

    print_section("ELLIPTIC XGBOOST BASELINE")

    print(f"XGBoost version: {xgb.__version__}")
    print(f"Random seed: {SEED}")
    print(f"Feature configuration: {FEATURE_KEY}")

    # 1. Load data
    data = load_processed_data()

    # 2. Prepare train and validation data
    (
        X_train,
        y_train,
        X_val,
        y_val
    ) = prepare_training_data(data)

    # 3. Handle class imbalance
    scale_pos_weight = calculate_scale_pos_weight(
        y_train
    )

    # 4. Train model
    model, training_time = train_model(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    scale_pos_weight=scale_pos_weight
)

    # 5. Generate probabilities for all nodes
    probabilities, scoring_time = generate_predictions(
        model,
        data["X"]
    )

    # 6. Evaluate validation and test
    metrics = evaluate_model(
        model,
        data,
        probabilities
    )

    # 7. Save results
    save_results(
    model=model,
    probabilities=probabilities,
    metrics=metrics,
    training_time=training_time,
    scoring_time=scoring_time
)

    print_section("XGBOOST BASELINE COMPLETED")

    print("All-node predictions are ready.")
    print(
        PREDICTION_DIR
        / "baseline_xgb_probs.npy"
    )


if __name__ == "__main__":
    main()

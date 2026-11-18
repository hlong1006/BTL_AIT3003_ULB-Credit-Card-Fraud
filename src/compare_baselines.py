
from pathlib import Path
import json

import numpy as np
import pandas as pd


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TABLE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
)

XGB_METRICS_PATH = (
    TABLE_DIR / "baseline_xgb_metrics.json"
)

IF_METRICS_PATH = (
    TABLE_DIR / "isolation_forest_metrics.json"
)

AE_METRICS_PATH = (
    TABLE_DIR / "autoencoder_metrics.json"
)


# =========================================================
# 2. MODEL CONFIGURATION
# =========================================================

MODEL_CONFIGS = [
    {
        "name": "XGBoost",
        "path": XGB_METRICS_PATH,

        "threshold_method": (
            "Fixed threshold = 0.5"
        ),

        "score_direction": (
            "Higher probability = more illicit"
        )
    },

    {
        "name": "Isolation Forest",
        "path": IF_METRICS_PATH,

        "threshold_method": (
            "Validation F1 maximization"
        ),

        "score_direction": (
            "Original score_samples direction; "
            "selected using validation labels"
        )
    },

    {
        "name": "Autoencoder",
        "path": AE_METRICS_PATH,

        "threshold_method": (
            "Validation F1 maximization"
        ),

        "score_direction": (
            "Negative reconstruction error; "
            "selected using validation labels"
        )
    }
]


SPLITS = [
    "validation",
    "test"
]


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


def load_json(path):
    """
    Load a JSON metrics file.

    Return a dictionary containing model metadata
    and validation/test metrics.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid metrics format: {path}"
        )

    return data


def get_value(metrics, key, default=np.nan):
    """
    Safely retrieve a metric value.
    """

    value = metrics.get(key, default)

    if value is None:
        return default

    return value


# =========================================================
# 4. LOAD ALL MODEL METRICS
# =========================================================

def load_all_metrics():
    """
    Load metrics from XGBoost, Isolation Forest
    and Autoencoder.
    """

    print_section("1. LOAD MODEL METRICS")

    models = {}

    for config in MODEL_CONFIGS:

        name = config["name"]
        path = config["path"]

        print(f"\nLoading {name}...")

        metrics_data = load_json(path)

        # Check required splits
        for split in SPLITS:

            if split not in metrics_data:
                raise ValueError(
                    f"{name} is missing '{split}' metrics."
                )

            if not isinstance(
                metrics_data[split],
                dict
            ):
                raise ValueError(
                    f"Invalid {split} metrics for {name}."
                )

        models[name] = {
            "config": config,
            "metadata": metrics_data,
            "metrics": {
                split: metrics_data[split]
                for split in SPLITS
            }
        }

        print(f"[PASS] {name}")
        print(f"       Source: {path.name}")

        if "training_time_seconds" in metrics_data:
            print(
                "       Training time: "
                f"{metrics_data['training_time_seconds']:.2f}s"
            )

        else:
            print(
                "       Training time: Not recorded"
            )

    return models


# =========================================================
# 5. BUILD MODEL COMPARISON TABLE
# =========================================================

def build_comparison_table(models):
    """
    Build a standardized comparison table.

    Each model has two rows:
        - Validation
        - Test

    Ranking metrics:
        - PR-AUC (Average Precision)
        - PR-AUC (Trapezoidal)
        - ROC-AUC

    Threshold-dependent metrics:
        - Precision
        - Recall
        - F1
        - TP, FP, TN, FN
    """

    print_section("2. BUILD MODEL COMPARISON TABLE")

    rows = []

    for model_name, model_data in models.items():

        config = model_data["config"]

        metadata = model_data["metadata"]

        for split in SPLITS:

            metrics = model_data["metrics"][split]

            tp = int(get_value(metrics, "tp", 0))
            fp = int(get_value(metrics, "fp", 0))
            tn = int(get_value(metrics, "tn", 0))
            fn = int(get_value(metrics, "fn", 0))

            n_samples = int(
                get_value(metrics, "n_samples", 0)
            )

            n_positive = int(
                get_value(metrics, "n_positive", 0)
            )

            n_negative = int(
                get_value(metrics, "n_negative", 0)
            )

            positive_rate = get_value(
                metrics,
                "positive_rate"
            )

            # Fallback if positive_rate is unavailable
            if pd.isna(positive_rate) and n_samples > 0:
                positive_rate = n_positive / n_samples

            pr_auc = get_value(
                metrics,
                "pr_auc"
            )

            # AP divided by positive prevalence.
            # Useful for interpreting AP across splits
            # with different class distributions.
            if (
                not pd.isna(pr_auc)
                and positive_rate > 0
            ):
                ap_lift = pr_auc / positive_rate
            else:
                ap_lift = np.nan

            row = {
                "model": model_name,

                "split": split.capitalize(),

                "threshold_method": (
                    config["threshold_method"]
                ),

                "score_direction": (
                    config["score_direction"]
                ),

                "n_samples": n_samples,

                "n_positive": n_positive,

                "n_negative": n_negative,

                "positive_rate": positive_rate,

                "pr_auc_ap": pr_auc,

                "pr_auc_trapezoid": get_value(
                    metrics,
                    "pr_auc_trapezoid"
                ),

                "ap_over_prevalence": ap_lift,

                "roc_auc": get_value(
                    metrics,
                    "roc_auc"
                ),

                "threshold": get_value(
                    metrics,
                    "threshold"
                ),

                "precision": get_value(
                    metrics,
                    "precision"
                ),

                "recall": get_value(
                    metrics,
                    "recall"
                ),

                "f1": get_value(
                    metrics,
                    "f1"
                ),

                "tp": tp,

                "fp": fp,

                "tn": tn,

                "fn": fn,

                "alerts": tp + fp,

                "training_time_seconds": get_value(
                    metadata,
                    "training_time_seconds"
                ),

                "scoring_time_seconds": get_value(
                    metadata,
                    "scoring_time_seconds"
                )
            }

            rows.append(row)

    comparison_df = pd.DataFrame(rows)

    return comparison_df


# =========================================================
# 6. BUILD TEST-ONLY COMPARISON TABLE
# =========================================================

def build_test_comparison(comparison_df):
    """
    Extract test results only.

    Sort models by PR-AUC (AP) descending.
    """

    print_section("3. BUILD TEST COMPARISON")

    test_df = comparison_df[
        comparison_df["split"] == "Test"
    ].copy()

    test_df = test_df.sort_values(
        by="pr_auc_ap",
        ascending=False
    ).reset_index(drop=True)

    test_df.insert(
        0,
        "ranking",
        np.arange(1, len(test_df) + 1)
    )

    return test_df


# =========================================================
# 7. BUILD RUNTIME COMPARISON
# =========================================================

def build_runtime_comparison(models):
    """
    Build a separate computational cost table.

    Missing runtime measurements are kept as NaN.
    They are never replaced with zero.
    """

    print_section("4. BUILD RUNTIME COMPARISON")

    rows = []

    for model_name, model_data in models.items():

        metadata = model_data["metadata"]

        training_time = get_value(
            metadata,
            "training_time_seconds"
        )

        scoring_time = get_value(
            metadata,
            "scoring_time_seconds"
        )

        if (
            not pd.isna(training_time)
            and not pd.isna(scoring_time)
        ):

            total_time = (
                training_time + scoring_time
            )

        else:

            total_time = np.nan

        rows.append({
            "model": model_name,

            "training_time_seconds": training_time,

            "scoring_time_seconds": scoring_time,

            "total_time_seconds": total_time
        })

    return pd.DataFrame(rows)


# =========================================================
# 8. SAVE COMPARISON TABLES
# =========================================================

def save_comparison_tables(
    comparison_df,
    test_df,
    runtime_df
):
    """
    Save all benchmark tables to outputs/tables.
    """

    print_section("5. SAVE COMPARISON TABLES")

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Full comparison
    # -----------------------------------------------------

    full_path = (
        TABLE_DIR
        / "p0_model_comparison.csv"
    )

    comparison_df.to_csv(
        full_path,
        index=False
    )

    print(f"Saved: {full_path}")

    # -----------------------------------------------------
    # Test-only comparison
    # -----------------------------------------------------

    test_path = (
        TABLE_DIR
        / "p0_test_model_comparison.csv"
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    print(f"Saved: {test_path}")

    # -----------------------------------------------------
    # Runtime comparison
    # -----------------------------------------------------

    runtime_path = (
        TABLE_DIR
        / "p0_runtime_comparison.csv"
    )

    runtime_df.to_csv(
        runtime_path,
        index=False
    )

    print(f"Saved: {runtime_path}")


# =========================================================
# 9. PRINT SUMMARY
# =========================================================

def print_summary(
    comparison_df,
    test_df,
    runtime_df
):
    """
    Print model comparison summaries.
    """

    print_section("6. VALIDATION AND TEST RESULTS")

    columns = [
        "model",
        "split",
        "pr_auc_ap",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "threshold"
    ]

    print(
        comparison_df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # -----------------------------------------------------
    # Test ranking
    # -----------------------------------------------------

    print_section("7. TEST MODEL RANKING BY PR-AUC")

    test_columns = [
        "ranking",
        "model",
        "pr_auc_ap",
        "roc_auc",
        "precision",
        "recall",
        "f1"
    ]

    print(
        test_df[test_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # -----------------------------------------------------
    # Runtime
    # -----------------------------------------------------

    print_section("8. COMPUTATIONAL COST")

    print(
        runtime_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print(
        "\nNote: NaN means runtime was not recorded, "
        "not zero."
    )


# =========================================================
# 10. VALIDATE COMPARISON RESULTS
# =========================================================

def validate_comparison(
    comparison_df,
    test_df
):
    """
    Perform final checks on the comparison tables.
    """

    print_section("9. VALIDATE COMPARISON")

    expected_models = {
        "XGBoost",
        "Isolation Forest",
        "Autoencoder"
    }

    actual_models = set(
        comparison_df["model"].unique()
    )

    if actual_models != expected_models:
        raise ValueError(
            f"Unexpected models: {actual_models}"
        )

    if len(comparison_df) != 6:
        raise ValueError(
            "Expected 6 rows: 3 models x 2 splits."
        )

    if len(test_df) != 3:
        raise ValueError(
            "Expected 3 test model results."
        )

    metric_columns = [
        "pr_auc_ap",
        "roc_auc",
        "precision",
        "recall",
        "f1"
    ]

    for column in metric_columns:

        if comparison_df[column].isna().any():
            raise ValueError(
                f"Missing metric values in {column}."
            )

        if not comparison_df[column].between(
            0,
            1
        ).all():
            raise ValueError(
                f"Metric values outside [0, 1]: {column}"
            )

    print("[PASS] All three models are included.")

    print("[PASS] Validation and test results are included.")

    print("[PASS] All comparison metrics are valid.")

    print("[PASS] Test ranking is complete.")


# =========================================================
# 11. MAIN
# =========================================================

def main():

    print_section("P0 MODEL BENCHMARK")

    # 1. Load metrics
    models = load_all_metrics()

    # 2. Build complete comparison
    comparison_df = build_comparison_table(
        models
    )

    # 3. Build test ranking
    test_df = build_test_comparison(
        comparison_df
    )

    # 4. Runtime comparison
    runtime_df = build_runtime_comparison(
        models
    )

    # 5. Validate
    validate_comparison(
        comparison_df,
        test_df
    )

    # 6. Save results
    save_comparison_tables(
        comparison_df,
        test_df,
        runtime_df
    )

    # 7. Print results
    print_summary(
        comparison_df,
        test_df,
        runtime_df
    )

    print_section("P0 BENCHMARK COMPLETED")

    print("All model comparison tables are ready.")


if __name__ == "__main__":
    main()

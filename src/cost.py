
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve

from src.metrics import (
    evaluate_binary_classifier
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

PREDICTION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "baseline_xgb_probs.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"


# =========================================================
# 2. COST CONFIGURATION
# =========================================================

# Cost ratios:
# C_FN / C_FP = 1, 5, 10, 50, 100

COST_RATIOS = [1, 5, 10, 50, 100]

# Normalize false-positive cost to 1
C_FP = 1.0

# Initial baseline threshold
BASELINE_THRESHOLD = 0.5


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

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# 4. LOAD DATA AND PREDICTIONS
# =========================================================

def load_data():
    """
    Load processed data and XGBoost predictions.

    The prediction vector must preserve the original
    transaction order.
    """

    print_section("1. LOAD DATA AND PREDICTIONS")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found: {DATA_PATH}"
        )

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTION_PATH}"
        )

    with np.load(
        DATA_PATH,
        allow_pickle=False
    ) as data:

        y = data["y"].copy()

        timestep = data["timestep"].copy()

        val_mask = data["val_mask"].copy()

        test_mask = data["test_mask"].copy()

    probabilities = np.load(
        PREDICTION_PATH,
        allow_pickle=False
    )

    # -----------------------------------------------------
    # Validate dimensions
    # -----------------------------------------------------

    n = len(y)

    if probabilities.shape != (n,):
        raise ValueError(
            f"Prediction shape mismatch. "
            f"Expected {(n,)}, got {probabilities.shape}"
        )

    if len(val_mask) != n or len(test_mask) != n:
        raise ValueError(
            "Mask dimensions do not match labels."
        )

    # -----------------------------------------------------
    # Validate prediction values
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Validate masks
    # -----------------------------------------------------

    if np.any(val_mask & test_mask):
        raise ValueError(
            "Validation and test masks overlap."
        )

    if np.any(y[val_mask] == -1):
        raise ValueError(
            "Validation mask contains unknown labels."
        )

    if np.any(y[test_mask] == -1):
        raise ValueError(
            "Test mask contains unknown labels."
        )

    print(f"Total nodes: {n:,}")

    print(
        f"Prediction shape: {probabilities.shape}"
    )

    print(
        f"Validation samples: {val_mask.sum():,}"
    )

    print(
        f"Test samples: {test_mask.sum():,}"
    )

    return {
        "y": y,
        "timestep": timestep,
        "probabilities": probabilities,
        "val_mask": val_mask,
        "test_mask": test_mask
    }


# =========================================================
# 5. BUILD THRESHOLD-COST CURVE
# =========================================================

def build_threshold_curve(y_true, y_score):
    """
    Calculate confusion matrix statistics for all
    relevant classification thresholds.

    Uses sorting and cumulative sums to avoid repeatedly
    scanning the entire prediction array.

    Returns:
        pandas.DataFrame
    """

    y_true = np.asarray(y_true).reshape(-1)

    y_score = np.asarray(
        y_score,
        dtype=np.float64
    ).reshape(-1)

    if len(y_true) != len(y_score):
        raise ValueError(
            "y_true and y_score have different lengths."
        )

    # Remove unknown labels if present
    labeled_mask = y_true != -1

    y_true = y_true[labeled_mask]
    y_score = y_score[labeled_mask]

    if len(y_true) == 0:
        raise ValueError(
            "No labeled samples available."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(
            "Labels must be binary after removing unknown."
        )

    if not np.isfinite(y_score).all():
        raise ValueError(
            "Scores contain NaN or Inf."
        )

    if np.any(
        (y_score < 0)
        | (y_score > 1)
    ):
        raise ValueError(
            "Scores must be in [0, 1]."
        )

    # -----------------------------------------------------
    # Sort scores in ascending order
    # -----------------------------------------------------

    sorted_indices = np.argsort(
        y_score,
        kind="mergesort"
    )

    sorted_scores = y_score[sorted_indices]

    sorted_y = y_true[sorted_indices]

    # -----------------------------------------------------
    # Generate all candidate thresholds
    # -----------------------------------------------------

    thresholds = np.unique(
        np.concatenate([
            np.array([0.0]),
            y_score,
            np.array([1.0])
        ])
    )

    # Position of the first score >= threshold
    indices = np.searchsorted(
        sorted_scores,
        thresholds,
        side="left"
    )

    # -----------------------------------------------------
    # Calculate cumulative positive counts
    # -----------------------------------------------------

    positive_prefix = np.concatenate([
        np.array([0], dtype=np.int64),
        np.cumsum(
            sorted_y == 1,
            dtype=np.int64
        )
    ])

    total_positive = int(
        (y_true == 1).sum()
    )

    total_negative = int(
        (y_true == 0).sum()
    )

    n = len(y_true)

    # -----------------------------------------------------
    # Confusion matrix for every threshold
    # -----------------------------------------------------

    tp = total_positive - positive_prefix[indices]

    predicted_positive = n - indices

    fp = predicted_positive - tp

    fn = total_positive - tp

    tn = total_negative - fp

    # -----------------------------------------------------
    # Classification metrics
    # -----------------------------------------------------

    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(
            tp,
            dtype=np.float64
        ),
        where=(tp + fp) > 0
    )

    recall = np.divide(
        tp,
        total_positive,
        out=np.zeros_like(
            tp,
            dtype=np.float64
        ),
        where=total_positive > 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(
            precision
        ),
        where=(precision + recall) > 0
    )

    alerts = tp + fp

    curve = pd.DataFrame({
        "threshold": thresholds,

        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "alerts": alerts
    })

    return curve


# =========================================================
# 6. CALCULATE COST
# =========================================================

def calculate_cost(
    fn,
    fp,
    c_fn,
    c_fp=C_FP
):
    """
    Calculate total operational cost.

    Cost = C_FN * FN + C_FP * FP
    """

    return (
        c_fn * fn
        + c_fp * fp
    )


# =========================================================
# 7. OPTIMIZE THRESHOLD ON VALIDATION
# =========================================================

def optimize_threshold(
    val_curve,
    cost_ratio
):
    """
    Select the threshold minimizing validation cost.

    Tie-breaking:
        1. Higher recall.
        2. Higher precision.
        3. Lower threshold.

    Test data is never used here.
    """

    c_fn = float(cost_ratio)
    c_fp = C_FP

    curve = val_curve.copy()

    curve["cost"] = calculate_cost(
        fn=curve["fn"].to_numpy(),
        fp=curve["fp"].to_numpy(),
        c_fn=c_fn,
        c_fp=c_fp
    )

    min_cost = curve["cost"].min()

    # Select thresholds with minimum cost
    candidates = curve[
        curve["cost"] == min_cost
    ].copy()

    # Explicit tie-breaking
    candidates = candidates.sort_values(
        by=[
            "recall",
            "precision",
            "threshold"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )

    best = candidates.iloc[0]

    return {
        "threshold": float(best["threshold"]),

        "cost": float(best["cost"]),

        "tp": int(best["tp"]),
        "fp": int(best["fp"]),
        "tn": int(best["tn"]),
        "fn": int(best["fn"]),

        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),

        "alerts": int(best["alerts"]),

        "c_fn": c_fn,
        "c_fp": c_fp
    }


# =========================================================
# 8. EVALUATE A SELECTED THRESHOLD
# =========================================================

def evaluate_selected_threshold(
    y_true,
    y_score,
    threshold,
    cost_ratio
):
    """
    Evaluate a fixed threshold and calculate cost.
    """

    metrics = evaluate_binary_classifier(
        y_true=y_true,
        y_score=y_score,
        threshold=threshold
    )

    cost = calculate_cost(
        fn=metrics["fn"],
        fp=metrics["fp"],
        c_fn=cost_ratio,
        c_fp=C_FP
    )

    result = {
        **metrics,

        "cost_ratio": cost_ratio,

        "cost": float(cost),

        "cost_per_sample": float(
            cost / metrics["n_samples"]
        ),

        "alerts": (
            metrics["tp"]
            + metrics["fp"]
        )
    }

    return result


# =========================================================
# 9. RUN COST OPTIMIZATION
# =========================================================

def run_cost_optimization(data):
    """
    Optimize thresholds on validation and evaluate
    the selected thresholds on test.
    """

    print_section("2. COST-SENSITIVE THRESHOLD OPTIMIZATION")

    y = data["y"]
    probabilities = data["probabilities"]

    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    # Extract labeled validation samples
    y_val = y[val_mask]

    score_val = probabilities[val_mask]

    # Extract labeled test samples
    y_test = y[test_mask]

    score_test = probabilities[test_mask]

    # -----------------------------------------------------
    # Build validation threshold curve
    # -----------------------------------------------------

    print("\nBuilding validation threshold curve...")

    val_curve = build_threshold_curve(
        y_val,
        score_val
    )

    print(
        f"Number of validation thresholds: "
        f"{len(val_curve):,}"
    )

    # -----------------------------------------------------
    # Build test threshold curve for analysis/plotting
    # -----------------------------------------------------

    print("\nBuilding test threshold curve...")

    test_curve = build_threshold_curve(
        y_test,
        score_test
    )

    print(
        f"Number of test thresholds: "
        f"{len(test_curve):,}"
    )

    # -----------------------------------------------------
    # Optimize each cost ratio on validation
    # -----------------------------------------------------

    rows = []

    selected_thresholds = {}

    for ratio in COST_RATIOS:

        print("\n" + "-" * 65)

        print(
            f"Cost ratio C_FN / C_FP = {ratio}"
        )

        # Select threshold ONLY using validation
        best = optimize_threshold(
            val_curve=val_curve,
            cost_ratio=ratio
        )

        threshold = best["threshold"]

        selected_thresholds[ratio] = threshold

        # Validation evaluation
        val_result = evaluate_selected_threshold(
            y_true=y_val,
            y_score=score_val,
            threshold=threshold,
            cost_ratio=ratio
        )

        # Test evaluation using the same threshold
        test_result = evaluate_selected_threshold(
            y_true=y_test,
            y_score=score_test,
            threshold=threshold,
            cost_ratio=ratio
        )

        row = {
            "cost_ratio_fn_fp": ratio,

            "c_fn": ratio,
            "c_fp": C_FP,

            "optimal_threshold": threshold,

            "val_cost": val_result["cost"],
            "val_cost_per_sample": (
                val_result["cost_per_sample"]
            ),

            "val_precision": val_result["precision"],
            "val_recall": val_result["recall"],
            "val_f1": val_result["f1"],

            "val_tp": val_result["tp"],
            "val_fp": val_result["fp"],
            "val_tn": val_result["tn"],
            "val_fn": val_result["fn"],

            "val_alerts": val_result["alerts"],

            "test_cost": test_result["cost"],
            "test_cost_per_sample": (
                test_result["cost_per_sample"]
            ),

            "test_precision": test_result["precision"],
            "test_recall": test_result["recall"],
            "test_f1": test_result["f1"],

            "test_tp": test_result["tp"],
            "test_fp": test_result["fp"],
            "test_tn": test_result["tn"],
            "test_fn": test_result["fn"],

            "test_alerts": test_result["alerts"]
        }

        rows.append(row)

        print(f"Optimal threshold: {threshold:.6f}")

        print("\nValidation:")
        print(f"Cost: {val_result['cost']:,.2f}")
        print(
            f"Precision: {val_result['precision']:.4f}"
        )
        print(
            f"Recall   : {val_result['recall']:.4f}"
        )
        print(
            f"F1       : {val_result['f1']:.4f}"
        )
        print(
            f"Alerts   : {val_result['alerts']:,}"
        )

        print("\nTest:")
        print(f"Cost: {test_result['cost']:,.2f}")
        print(
            f"Precision: {test_result['precision']:.4f}"
        )
        print(
            f"Recall   : {test_result['recall']:.4f}"
        )
        print(
            f"F1       : {test_result['f1']:.4f}"
        )
        print(
            f"Alerts   : {test_result['alerts']:,}"
        )

    summary = pd.DataFrame(rows)

    return {
        "summary": summary,
        "val_curve": val_curve,
        "test_curve": test_curve,
        "selected_thresholds": selected_thresholds
    }


# =========================================================
# 10. BASELINE COST AT THRESHOLD 0.5
# =========================================================

def evaluate_baseline_cost(data):
    """
    Calculate cost at the original threshold 0.5.

    This is a reference for comparison.
    """

    print_section("3. BASELINE COST AT THRESHOLD 0.5")

    y = data["y"]
    scores = data["probabilities"]

    results = []

    for split_name, mask in [
        ("Validation", data["val_mask"]),
        ("Test", data["test_mask"])
    ]:

        metrics = evaluate_binary_classifier(
            y_true=y[mask],
            y_score=scores[mask],
            threshold=BASELINE_THRESHOLD
        )

        for ratio in COST_RATIOS:

            cost = calculate_cost(
                fn=metrics["fn"],
                fp=metrics["fp"],
                c_fn=ratio
            )

            results.append({
                "split": split_name,

                "cost_ratio_fn_fp": ratio,

                "threshold": BASELINE_THRESHOLD,

                "cost": cost,

                "cost_per_sample": (
                    cost / metrics["n_samples"]
                ),

                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],

                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"]
            })

    baseline_df = pd.DataFrame(results)

    print(baseline_df.to_string(index=False))

    return baseline_df


# =========================================================
# 11. PLOT COST VS THRESHOLD
# =========================================================

def plot_cost_vs_threshold(val_curve):
    """
    Plot validation cost per sample against threshold
    for all cost ratios.
    """

    print("\nPlotting cost vs threshold...")

    plt.figure(figsize=(10, 6))

    for ratio in COST_RATIOS:

        costs = calculate_cost(
            fn=val_curve["fn"].to_numpy(),
            fp=val_curve["fp"].to_numpy(),
            c_fn=ratio
        )

        costs_per_sample = (
            costs / (
                val_curve["tp"]
                + val_curve["fp"]
                + val_curve["tn"]
                + val_curve["fn"]
            )
        )

        plt.plot(
            val_curve["threshold"],
            costs_per_sample,
            label=f"FN/FP = {ratio}"
        )

    plt.xlabel("Threshold")
    plt.ylabel("Validation cost per sample")

    plt.title("Cost vs Classification Threshold (Validation)")

    plt.xlim(0, 1)

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "cost_vs_threshold_validation.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


# =========================================================
# 12. PLOT PRECISION-RECALL CURVES
# =========================================================

def plot_precision_recall(data):
    """
    Plot validation and test Precision-Recall curves.

    Test curve is for reporting, not threshold selection.
    """

    print("\nPlotting Precision-Recall curves...")

    plt.figure(figsize=(9, 7))

    for split_name, mask in [
        ("Validation", data["val_mask"]),
        ("Test", data["test_mask"])
    ]:

        y_true = data["y"][mask]

        y_score = data["probabilities"][mask]

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_score
        )

        metrics = evaluate_binary_classifier(
            y_true,
            y_score,
            threshold=BASELINE_THRESHOLD
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{split_name} "
                f"(AP={metrics['pr_auc']:.4f})"
            )
        )

        # Random classifier reference
        prevalence = np.mean(y_true == 1)

        plt.axhline(
            prevalence,
            linestyle="--",
            alpha=0.4,
            label=f"{split_name} prevalence"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curves")

    plt.xlim(0, 1)
    plt.ylim(0, 1.02)

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "precision_recall_curves.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


# =========================================================
# 13. PLOT RECALL VS NUMBER OF ALERTS
# =========================================================


def plot_recall_vs_alerts(
    test_curve,
    selected_thresholds,
    y_test,
    score_test
):
    """
    Plot test recall according to the number of
    transactions that need investigation.

    Highlight thresholds selected on validation.
    """

    print("\nPlotting recall vs number of alerts...")

    # Test operating curve
    curve = test_curve.sort_values("alerts")

    plt.figure(figsize=(10, 6))

    plt.plot(
        curve["alerts"],
        curve["recall"],
        linewidth=2,
        label="Test operating curve"
    )

    # Highlight validation-selected thresholds
    for ratio, threshold in selected_thresholds.items():

        # Evaluate directly on test using the selected threshold
        result = evaluate_binary_classifier(
            y_true=y_test,
            y_score=score_test,
            threshold=threshold
        )

        # Number of transactions requiring investigation
        alerts = result["tp"] + result["fp"]

        plt.scatter(
            alerts,
            result["recall"],
            s=55,
            label=(
                f"FN/FP = {ratio}, "
                f"threshold = {threshold:.3f}"
            )
        )

    plt.xlabel("Number of alerts requiring investigation")
    plt.ylabel("Recall of illicit transactions")

    plt.title("Recall vs Investigation Workload (Test)")

    plt.xlim(left=0)
    plt.ylim(0, 1.02)

    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "recall_vs_alerts_test.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")



# =========================================================
# 14. PLOT COST COMPARISON
# =========================================================

def plot_cost_comparison(summary):
    """
    Compare validation and test cost per 1,000 labeled
    transactions at validation-selected thresholds.
    """

    print("\nPlotting cost comparison...")

    ratios = summary["cost_ratio_fn_fp"]

    val_cost = (
        summary["val_cost_per_sample"] * 1000
    )

    test_cost = (
        summary["test_cost_per_sample"] * 1000
    )

    x = np.arange(len(ratios))

    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width / 2,
        val_cost,
        width,
        label="Validation"
    )

    plt.bar(
        x + width / 2,
        test_cost,
        width,
        label="Test"
    )

    plt.xticks(
        x,
        [str(ratio) for ratio in ratios]
    )

    plt.xlabel("Cost ratio C_FN / C_FP")

    plt.ylabel("Cost per 1,000 labeled transactions")

    plt.title(
        "Validation vs Test Cost at Selected Thresholds"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "cost_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


# =========================================================
# 15. SAVE RESULTS
# =========================================================

def save_results(
    summary,
    baseline_df,
    val_curve,
    test_curve
):
    """
    Save tables and threshold curves.
    """

    print_section("4. SAVE COST OPTIMIZATION RESULTS")

    create_output_directories()

    # -----------------------------------------------------
    # Main summary table
    # -----------------------------------------------------

    summary_path = (
        TABLE_DIR
        / "cost_threshold_results.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print(f"Saved: {summary_path}")

    # -----------------------------------------------------
    # Baseline cost table
    # -----------------------------------------------------

    baseline_path = (
        TABLE_DIR
        / "baseline_cost_results.csv"
    )

    baseline_df.to_csv(
        baseline_path,
        index=False
    )

    print(f"Saved: {baseline_path}")

    # -----------------------------------------------------
    # Threshold curves
    # -----------------------------------------------------

    val_curve_path = (
        TABLE_DIR
        / "validation_threshold_curve.csv"
    )

    test_curve_path = (
        TABLE_DIR
        / "test_threshold_curve.csv"
    )

    val_curve.to_csv(
        val_curve_path,
        index=False
    )

    test_curve.to_csv(
        test_curve_path,
        index=False
    )

    print(f"Saved: {val_curve_path}")
    print(f"Saved: {test_curve_path}")


# =========================================================
# 16. MAIN
# =========================================================

def main():

    print_section("ELLIPTIC COST-SENSITIVE OPTIMIZATION")

    create_output_directories()

    # 1. Load data
    data = load_data()

    # 2. Optimize thresholds using validation
    optimization_results = run_cost_optimization(
        data
    )

    summary = optimization_results["summary"]

    val_curve = optimization_results["val_curve"]

    test_curve = optimization_results["test_curve"]

    selected_thresholds = (
        optimization_results["selected_thresholds"]
    )

    # 3. Calculate baseline cost at threshold 0.5
    baseline_df = evaluate_baseline_cost(data)

    # 4. Generate visualizations
    plot_cost_vs_threshold(
        val_curve
    )

    plot_precision_recall(
        data
    )

    plot_recall_vs_alerts(
        test_curve,
        selected_thresholds,
        data["y"][data["test_mask"]],
        data["probabilities"][data["test_mask"]]
    )

    plot_cost_comparison(
        summary
    )

    # 5. Save results
    save_results(
        summary,
        baseline_df,
        val_curve,
        test_curve
    )

    # 6. Final report
    print_section("COST OPTIMIZATION SUMMARY")

    display_columns = [
        "cost_ratio_fn_fp",
        "optimal_threshold",
        "val_cost",
        "val_precision",
        "val_recall",
        "val_alerts",
        "test_cost",
        "test_precision",
        "test_recall",
        "test_alerts"
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print_section("COST OPTIMIZATION COMPLETED")

    print("All thresholds were selected using validation.")
    print("Test was used only for final evaluation.")


if __name__ == "__main__":
    main()

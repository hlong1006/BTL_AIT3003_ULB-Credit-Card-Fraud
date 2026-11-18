
from pathlib import Path
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
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

INPUT_DIM = 165
LATENT_DIM = 32

HIDDEN_DIMS = [128, 64]

LEARNING_RATE = 0.001

BATCH_SIZE = 512

MAX_EPOCHS = 100

PATIENCE = 10

MIN_DELTA = 1e-6

THRESHOLD_METRIC = "f1"

SCORE_BATCH_SIZE = 4096


# =========================================================
# 3. REPRODUCIBILITY AND DEVICE
# =========================================================

def seed_everything(seed=SEED):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """
    Automatically select CUDA or CPU.
    """

    if torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")

    print(f"Selected device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return device


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
# 4. LOAD AND STANDARDIZE DATA
# =========================================================

def load_processed_data():
    """
    Load processed Elliptic data.

    StandardScaler is fitted ONLY on train_time_mask.

    All training-time nodes, including unknown nodes,
    are retained for unsupervised learning.
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

        timestep = data["timestep"].copy()

        train_time_mask = (
            data["train_time_mask"].copy()
        )

        val_time_mask = (
            data["val_time_mask"].copy()
        )

        test_time_mask = (
            data["test_time_mask"].copy()
        )

        val_mask = data["val_mask"].copy()

        test_mask = data["test_mask"].copy()

    # -----------------------------------------------------
    # Validate dimensions
    # -----------------------------------------------------

    n = len(y)

    if X.shape != (n, INPUT_DIM):
        raise ValueError(
            f"Unexpected feature shape: {X.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Features contain NaN or Inf."
        )

    # -----------------------------------------------------
    # Validate temporal masks
    # -----------------------------------------------------

    if np.any(train_time_mask & val_time_mask):
        raise ValueError(
            "Training and validation time masks overlap."
        )

    if np.any(train_time_mask & test_time_mask):
        raise ValueError(
            "Training and test time masks overlap."
        )

    if np.any(val_time_mask & test_time_mask):
        raise ValueError(
            "Validation and test time masks overlap."
        )

    # -----------------------------------------------------
    # Validate supervised masks
    # -----------------------------------------------------

    if np.any(val_mask & ~val_time_mask):
        raise ValueError(
            "val_mask must be a subset of val_time_mask."
        )

    if np.any(test_mask & ~test_time_mask):
        raise ValueError(
            "test_mask must be a subset of test_time_mask."
        )

    if np.any(y[val_mask] == -1):
        raise ValueError(
            "Validation mask contains unknown labels."
        )

    if np.any(y[test_mask] == -1):
        raise ValueError(
            "Test mask contains unknown labels."
        )

    # -----------------------------------------------------
    # Fit scaler on training data only
    # -----------------------------------------------------

    print("\nFitting StandardScaler...")

    scaler = StandardScaler()

    X_train_raw = X[train_time_mask]

    scaler.fit(X_train_raw)

    # Transform all nodes using the training scaler
    X_scaled = scaler.transform(X).astype(
        np.float32
    )

    if not np.isfinite(X_scaled).all():
        raise ValueError(
            "Scaled features contain NaN or Inf."
        )

    print("Scaler fitted on training nodes only.")

    print(f"Feature shape: {X_scaled.shape}")

    print("\nTemporal split:")

    print(
        f"Training nodes: {train_time_mask.sum():,}"
    )

    print(
        f"Validation-time nodes: {val_time_mask.sum():,}"
    )

    print(
        f"Validation labeled nodes: {val_mask.sum():,}"
    )

    print(
        f"Test labeled nodes: {test_mask.sum():,}"
    )

    print("\nTraining composition:")

    print(
        f"Labeled: "
        f"{np.sum(train_time_mask & (y >= 0)):,}"
    )

    print(
        f"Unknown: "
        f"{np.sum(train_time_mask & (y == -1)):,}"
    )

    return {
        "X": X_scaled,
        "y": y,
        "timestep": timestep,

        "train_time_mask": train_time_mask,
        "val_time_mask": val_time_mask,
        "test_time_mask": test_time_mask,

        "val_mask": val_mask,
        "test_mask": test_mask,

        "scaler": scaler
    }


# =========================================================
# 5. AUTOENCODER ARCHITECTURE
# =========================================================

class Autoencoder(nn.Module):
    """
    Fully connected undercomplete Autoencoder.

    Architecture:

        165 -> 128 -> 64 -> 32
             -> 64 -> 128 -> 165

    Latent dimension = 32.
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM
    ):

        super().__init__()

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),

            nn.Linear(hidden_dims[1], latent_dim)
        )

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),

            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),

            nn.Linear(hidden_dims[0], input_dim)
        )

    def forward(self, x):
        """
        Encode and reconstruct the input.
        """

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed


# =========================================================
# 6. PREPARE DATALOADERS
# =========================================================

def prepare_dataloaders(data, device):
    """
    Build training and validation DataLoaders.

    Training:
        All nodes from timestep 1-28.

    Validation:
        All nodes from timestep 29-34,
        including unknown nodes.

    No labels are used in either DataLoader.
    """

    print_section("2. PREPARE DATALOADERS")

    X = data["X"]

    train_mask = data["train_time_mask"]

    val_time_mask = data["val_time_mask"]

    X_train = X[train_mask]

    X_val = X[val_time_mask]

    train_dataset = TensorDataset(
        torch.from_numpy(
            np.ascontiguousarray(X_train)
        )
    )

    val_dataset = TensorDataset(
        torch.from_numpy(
            np.ascontiguousarray(X_val)
        )
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        drop_last=False,

        num_workers=0,

        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        drop_last=False,

        num_workers=0,

        pin_memory=pin_memory
    )

    print(f"Training samples: {len(train_dataset):,}")

    print(f"Validation samples: {len(val_dataset):,}")

    print(f"Batch size: {BATCH_SIZE}")

    print(
        "\n[PASS] Unknown nodes included in unsupervised training."
    )

    print(
        "[PASS] Validation reconstruction loss uses all validation-time nodes."
    )

    print(
        "[PASS] No labels are passed to the DataLoaders."
    )

    return train_loader, val_loader


# =========================================================
# 7. CALCULATE EPOCH LOSS
# =========================================================

def calculate_epoch_loss(
    model,
    loader,
    criterion,
    device
):
    """
    Calculate mean reconstruction loss over a dataset.
    """

    model.eval()

    total_loss = 0.0

    total_samples = 0

    with torch.no_grad():

        for (batch,) in loader:

            batch = batch.to(
                device,
                non_blocking=True
            )

            reconstructed = model(batch)

            loss = criterion(
                reconstructed,
                batch
            )

            batch_size = batch.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Cannot calculate loss on an empty dataset."
        )

    return total_loss / total_samples


# =========================================================
# 8. TRAIN AUTOENCODER WITH EARLY STOPPING
# =========================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device
):
    """
    Train Autoencoder using reconstruction MSE.

    Early stopping is based on validation reconstruction loss.

    Validation labels are NOT used.
    """

    print_section("3. TRAIN AUTOENCODER")

    criterion = nn.MSELoss(
        reduction="mean"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    best_epoch = 0

    best_state = None

    epochs_without_improvement = 0

    history = []

    print(f"Maximum epochs: {MAX_EPOCHS}")

    print(f"Early stopping patience: {PATIENCE}")

    print(f"Learning rate: {LEARNING_RATE}")

    print("\nTraining started...\n")

    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        total_train_loss = 0.0

        total_samples = 0

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        for (batch,) in train_loader:

            batch = batch.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            reconstructed = model(batch)

            loss = criterion(
                reconstructed,
                batch
            )

            loss.backward()

            optimizer.step()

            batch_size = batch.size(0)

            total_train_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        train_loss = (
            total_train_loss / total_samples
        )

        # -------------------------------------------------
        # Validation reconstruction loss
        # -------------------------------------------------

        val_loss = calculate_epoch_loss(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        # -------------------------------------------------
        # Store history
        # -------------------------------------------------

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        improved = (
            val_loss < best_val_loss - MIN_DELTA
        )

        if improved:

            best_val_loss = val_loss

            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------

        if epoch == 1 or epoch % 10 == 0 or improved:

            print(
                f"Epoch [{epoch:03d}/{MAX_EPOCHS}] "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f}"
            )

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping at epoch {epoch}."
            )

            break

    training_time = (
        time.perf_counter() - start_time
    )

    # -----------------------------------------------------
    # Restore best checkpoint
    # -----------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            "Training failed to produce a valid model."
        )

    model.load_state_dict(best_state)

    model.to(device)

    print("\nTraining completed!")

    print(f"Best epoch: {best_epoch}")

    print(
        f"Best validation MSE: {best_val_loss:.6f}"
    )

    print(
        f"Training time: {training_time:.2f} seconds"
    )

    history_df = pd.DataFrame(history)

    return (
        model,
        best_epoch,
        best_val_loss,
        training_time,
        history_df
    )


# =========================================================
# 9. GENERATE RAW RECONSTRUCTION ERRORS
# =========================================================

def generate_reconstruction_errors(
    model,
    X,
    device
):
    """
    Generate raw reconstruction error for every node.

    Score:
        Mean squared error across all 165 features.

    Higher raw error = greater reconstruction difficulty.

    This function returns the original, non-negative
    reconstruction errors. It does NOT invert the score.
    """

    print_section("4. GENERATE RECONSTRUCTION ERRORS")

    model.eval()

    n = len(X)

    all_scores = []

    start_time = time.perf_counter()

    with torch.no_grad():

        for start in range(
            0,
            n,
            SCORE_BATCH_SIZE
        ):

            end = min(
                start + SCORE_BATCH_SIZE,
                n
            )

            batch = torch.from_numpy(
                np.ascontiguousarray(X[start:end])
            ).to(device)

            reconstructed = model(batch)

            errors = torch.mean(
                (batch - reconstructed) ** 2,
                dim=1
            )

            all_scores.append(
                errors.cpu().numpy()
            )

    raw_reconstruction_errors = np.concatenate(
        all_scores
    ).astype(np.float32)

    scoring_time = (
        time.perf_counter() - start_time
    )

    if raw_reconstruction_errors.shape != (n,):
        raise ValueError(
            "Unexpected reconstruction error shape."
        )

    if not np.isfinite(raw_reconstruction_errors).all():
        raise ValueError(
            "Reconstruction errors contain NaN or Inf."
        )

    if np.any(raw_reconstruction_errors < 0):
        raise ValueError(
            "Reconstruction errors must be non-negative."
        )

    print(
        f"Score shape: {raw_reconstruction_errors.shape}"
    )

    print(
        f"Minimum error: "
        f"{raw_reconstruction_errors.min():.8f}"
    )

    print(
        f"Maximum error: "
        f"{raw_reconstruction_errors.max():.8f}"
    )

    print(
        f"Mean error: "
        f"{raw_reconstruction_errors.mean():.8f}"
    )

    print(
        f"Scoring time: {scoring_time:.2f} seconds"
    )

    return raw_reconstruction_errors, scoring_time


# =========================================================
# 10. SELECT THRESHOLD ON VALIDATION
# =========================================================

def select_validation_threshold(
    y_val,
    detection_scores_val
):
    """
    Select threshold maximizing F1 on labeled validation nodes.

    Detection score convention:
        Higher score = more associated with illicit.

    The Autoencoder is trained without labels.
    Validation labels are used only for threshold selection.
    """

    print_section("5. SELECT THRESHOLD ON VALIDATION")

    y_val = np.asarray(
        y_val
    ).reshape(-1)

    scores = np.asarray(
        detection_scores_val,
        dtype=np.float64
    ).reshape(-1)

    if len(y_val) != len(scores):
        raise ValueError(
            "Labels and detection scores have different lengths."
        )

    if not np.isin(y_val, [0, 1]).all():
        raise ValueError(
            "Validation labels must be binary."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Detection scores contain NaN or Inf."
        )

    if len(np.unique(y_val)) < 2:
        raise ValueError(
            "Validation must contain both classes."
        )

    precision, recall, thresholds = (
        precision_recall_curve(
            y_val,
            scores,
            pos_label=1
        )
    )

    # Precision and recall contain one extra endpoint
    # without an associated threshold.
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

    # Select exact maximum-F1 candidates.
    candidate_indices = np.flatnonzero(
        f1_scores == max_f1
    )

    # Tie-breaking:
    # 1. Higher precision
    # 2. Higher threshold

    best_idx = candidate_indices[
        np.lexsort((
            -thresholds[candidate_indices],
            -precision[candidate_indices]
        ))[0]
    ]

    best_threshold = float(
        thresholds[best_idx]
    )

    print(
        f"Selected threshold: {best_threshold:.8f}"
    )

    print(
        f"Validation F1: {f1_scores[best_idx]:.6f}"
    )

    print(
        f"Validation Precision: {precision[best_idx]:.6f}"
    )

    print(
        f"Validation Recall: {recall[best_idx]:.6f}"
    )

    return best_threshold


# =========================================================
# 11. EVALUATE AUTOENCODER
# =========================================================

def evaluate_model(
    data,
    detection_scores,
    threshold
):
    """
    Evaluate validation and test using detection scores.

    Unknown labels are automatically excluded by
    evaluate_binary_classifier().
    """

    print_section("6. EVALUATE AUTOENCODER")

    y = data["y"]

    val_mask = data["val_mask"]

    test_mask = data["test_mask"]

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    print("\nVALIDATION RESULTS")

    val_metrics = evaluate_binary_classifier(
        y_true=y[val_mask],
        y_score=detection_scores[val_mask],
        threshold=threshold
    )

    print_metrics(val_metrics)

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    print("\nTEST RESULTS")

    test_metrics = evaluate_binary_classifier(
        y_true=y[test_mask],
        y_score=detection_scores[test_mask],
        threshold=threshold
    )

    print_metrics(test_metrics)

    return {
        "validation": val_metrics,
        "test": test_metrics
    }


# =========================================================
# 12. SAVE RESULTS
# =========================================================

def save_results(
    model,
    scaler,
    detection_scores,
    raw_reconstruction_errors,
    threshold,
    metrics,
    best_epoch,
    best_val_loss,
    training_time,
    scoring_time,
    history_df
):
    """
    Save:

        - Detection scores (direction adjusted)
        - Original reconstruction errors
        - Model checkpoint
        - StandardScaler
        - Metrics and metadata
        - Training history
    """

    print_section("7. SAVE AUTOENCODER RESULTS")

    create_output_directories()

    # -----------------------------------------------------
    # Validate score consistency
    # -----------------------------------------------------

    if detection_scores.shape != raw_reconstruction_errors.shape:
        raise ValueError(
            "Detection scores and raw reconstruction errors "
            "have different shapes."
        )

    if not np.allclose(
        detection_scores,
        -raw_reconstruction_errors,
        rtol=1e-5,
        atol=1e-7
    ):
        raise ValueError(
            "Detection scores do not match negative "
            "reconstruction errors."
        )

    # -----------------------------------------------------
    # Save detection scores
    # -----------------------------------------------------

    score_path = (
        PREDICTION_DIR
        / "autoencoder_scores.npy"
    )

    np.save(
        score_path,
        detection_scores
    )

    print(f"Detection scores saved: {score_path}")

    # -----------------------------------------------------
    # Save original reconstruction errors
    # -----------------------------------------------------

    raw_score_path = (
        PREDICTION_DIR
        / "autoencoder_reconstruction_errors.npy"
    )

    np.save(
        raw_score_path,
        raw_reconstruction_errors
    )

    print(
        f"Raw reconstruction errors saved: {raw_score_path}"
    )

    # -----------------------------------------------------
    # Save model checkpoint
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR
        / "autoencoder.pt"
    )

    # Store model weights on CPU for portability.
    cpu_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": cpu_state_dict,

        "input_dim": INPUT_DIM,

        "hidden_dims": HIDDEN_DIMS,

        "latent_dim": LATENT_DIM,

        "best_epoch": best_epoch,

        "best_val_loss": best_val_loss,

        "seed": SEED
    }

    torch.save(
        checkpoint,
        model_path
    )

    print(f"Model saved: {model_path}")

    # -----------------------------------------------------
    # Save scaler
    # -----------------------------------------------------

    scaler_path = (
        MODEL_DIR
        / "autoencoder_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print(f"Scaler saved: {scaler_path}")

    # -----------------------------------------------------
    # Save metrics JSON
    # -----------------------------------------------------

    metrics_path = (
        TABLE_DIR
        / "autoencoder_metrics.json"
    )

    results = {
        "algorithm": "Autoencoder",

        "seed": SEED,

        "feature_configuration": FEATURE_KEY,

        "input_dim": INPUT_DIM,

        "hidden_dims": HIDDEN_DIMS,

        "latent_dim": LATENT_DIM,

        "learning_rate": LEARNING_RATE,

        "batch_size": BATCH_SIZE,

        "max_epochs": MAX_EPOCHS,

        "patience": PATIENCE,

        "loss": "MSE",

        "optimizer": "Adam",

        "scaler": "StandardScaler",

        "scaler_fit": "train_time_mask_only",

        "training_labels_used": False,

        "early_stopping_monitor": (
            "reconstruction_mse_on_all_val_time_nodes"
        ),

        "threshold_selection": "maximize_validation_f1",

        "threshold": threshold,

        "raw_score_method": (
            "mean_squared_reconstruction_error"
        ),

        "score_method": (
            "negative_mean_squared_reconstruction_error"
        ),

        "score_direction": (
            "Higher detection scores are more associated "
            "with illicit labels; direction selected "
            "using validation labels."
        ),

        "best_epoch": best_epoch,

        "best_validation_reconstruction_loss": best_val_loss,

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
            "model": "Autoencoder",
            "split": "Validation",
            **metrics["validation"]
        },
        {
            "model": "Autoencoder",
            "split": "Test",
            **metrics["test"]
        }
    ])

    comparison_path = (
        TABLE_DIR
        / "autoencoder_results.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    print(f"Comparison saved: {comparison_path}")

    # -----------------------------------------------------
    # Save training history
    # -----------------------------------------------------

    history_path = (
        TABLE_DIR
        / "autoencoder_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False
    )

    print(f"Training history saved: {history_path}")


# =========================================================
# 13. MAIN
# =========================================================

def main():

    print_section("ELLIPTIC AUTOENCODER")

    # -----------------------------------------------------
    # 1. Reproducibility
    # -----------------------------------------------------

    seed_everything()

    # -----------------------------------------------------
    # 2. Device
    # -----------------------------------------------------

    device = get_device()

    # -----------------------------------------------------
    # 3. Create output directories
    # -----------------------------------------------------

    create_output_directories()

    # -----------------------------------------------------
    # 4. Load and standardize data
    # -----------------------------------------------------

    data = load_processed_data()

    # -----------------------------------------------------
    # 5. Prepare DataLoaders
    # -----------------------------------------------------

    train_loader, val_loader = prepare_dataloaders(
        data,
        device
    )

    # -----------------------------------------------------
    # 6. Build Autoencoder
    # -----------------------------------------------------

    print_section("BUILD AUTOENCODER")

    model = Autoencoder(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        latent_dim=LATENT_DIM
    ).to(device)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(model)

    print(
        f"\nTotal parameters: {total_parameters:,}"
    )

    # -----------------------------------------------------
    # 7. Train with early stopping
    # -----------------------------------------------------

    (
        model,
        best_epoch,
        best_val_loss,
        training_time,
        history_df
    ) = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # -----------------------------------------------------
    # 8. Generate original reconstruction errors
    # -----------------------------------------------------

    raw_reconstruction_errors, scoring_time = (
        generate_reconstruction_errors(
            model=model,
            X=data["X"],
            device=device
        )
    )

    # -----------------------------------------------------
    # 9. Convert into detection scores
    # -----------------------------------------------------

    # Empirical validation results indicate that
    # illicit transactions tend to have lower
    # reconstruction errors.
    #
    # Therefore, negate the raw reconstruction errors
    # so higher detection scores are more associated
    # with illicit transactions.
    #
    # This score direction is selected using validation,
    # never using test labels.

    detection_scores = -raw_reconstruction_errors

    print("\nScore direction adjustment:")

    print(
        "Raw reconstruction error: "
        "higher = greater reconstruction difficulty"
    )

    print(
        "Detection score: "
        "higher = more associated with illicit"
    )

    print(
        f"Detection score range: "
        f"[{detection_scores.min():.8f}, "
        f"{detection_scores.max():.8f}]"
    )

    # -----------------------------------------------------
    # 10. Select threshold using validation labels
    # -----------------------------------------------------

    val_mask = data["val_mask"]

    threshold = select_validation_threshold(
        y_val=data["y"][val_mask],

        detection_scores_val=(
            detection_scores[val_mask]
        )
    )

    # -----------------------------------------------------
    # 11. Evaluate
    # -----------------------------------------------------

    metrics = evaluate_model(
        data=data,

        detection_scores=detection_scores,

        threshold=threshold
    )

    # -----------------------------------------------------
    # 12. Save results
    # -----------------------------------------------------

    save_results(
        model=model,

        scaler=data["scaler"],

        detection_scores=detection_scores,

        raw_reconstruction_errors=raw_reconstruction_errors,

        threshold=threshold,

        metrics=metrics,

        best_epoch=best_epoch,

        best_val_loss=best_val_loss,

        training_time=training_time,

        scoring_time=scoring_time,

        history_df=history_df
    )

    # -----------------------------------------------------
    # 13. Final summary
    # -----------------------------------------------------

    print_section("AUTOENCODER COMPLETED")

    print("All-node detection scores are ready.")

    print(
        "Detection scores:",
        PREDICTION_DIR / "autoencoder_scores.npy"
    )

    print(
        "Raw reconstruction errors:",
        PREDICTION_DIR
        / "autoencoder_reconstruction_errors.npy"
    )


if __name__ == "__main__":
    main()

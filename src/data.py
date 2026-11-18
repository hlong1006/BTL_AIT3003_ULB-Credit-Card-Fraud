
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURES_PATH = RAW_DIR / "elliptic_txs_features.csv"
CLASSES_PATH = RAW_DIR / "elliptic_txs_classes.csv"
EDGES_PATH = RAW_DIR / "elliptic_txs_edgelist.csv"

OUTPUT_PATH = PROCESSED_DIR / "processed_data.npz"


# Temporal split
TRAIN_TIMESTEPS = np.arange(1, 29)
VAL_TIMESTEPS = np.arange(29, 35)
TEST_TIMESTEPS = np.arange(35, 50)


# Expected dataset dimensions
EXPECTED_NODES = 203769
EXPECTED_EDGES = 234355

NUM_LOCAL_FEATURES = 93
NUM_AGGREGATED_FEATURES = 72
NUM_FEATURES = 165


# Original labels -> numerical labels
LABEL_MAPPING = {
    "1": 1,          # Illicit
    "2": 0,          # Licit
    "unknown": -1    # Unknown
}


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def convert_to_int64(series, name):
    """
    Convert a pandas Series to int64.

    Raise an error if values are invalid or non-integer.
    """

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.isna().any():
        raise ValueError(
            f"{name} contains invalid or missing values."
        )

    values = numeric.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    if not np.equal(values, np.floor(values)).all():
        raise ValueError(
            f"{name} contains non-integer values."
        )

    return numeric.astype("int64")


def print_section(title):
    """
    Print a formatted section title.
    """

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# =========================================================
# 3. LOAD RAW DATA
# =========================================================

def load_raw_data():
    """
    Load the three original Elliptic CSV files.

    Features:
        No header.

    Classes:
        Header = txId, class.

    Edges:
        Header = txId1, txId2.
    """

    print_section("1. LOAD RAW DATA")

    # Check file existence
    required_files = [
        FEATURES_PATH,
        CLASSES_PATH,
        EDGES_PATH
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find file: {path}"
            )

        print(f"[FOUND] {path.name}")

    # Load CSV files
    features_df = pd.read_csv(
        FEATURES_PATH,
        header=None,
        low_memory=False
    )

    classes_df = pd.read_csv(
        CLASSES_PATH,
        header=0,
        low_memory=False
    )

    edges_df = pd.read_csv(
        EDGES_PATH,
        header=0,
        low_memory=False
    )

    # Normalize column names
    classes_df.columns = (
        classes_df.columns.astype(str).str.strip()
    )

    edges_df.columns = (
        edges_df.columns.astype(str).str.strip()
    )

    print("\nDataset shapes:")
    print("Features:", features_df.shape)
    print("Classes :", classes_df.shape)
    print("Edges   :", edges_df.shape)

    return features_df, classes_df, edges_df


# =========================================================
# 4. PREPROCESS FEATURE DATA
# =========================================================

def preprocess_features(features_df):
    """
    Extract transaction IDs, timesteps and features.

    Original feature CSV:
        Column 0       : txId
        Column 1       : timestep
        Columns 2-94   : 93 local features
        Columns 95-166 : 72 aggregated features

    The timestep column is NOT included in X.
    """

    print_section("2. PREPROCESS FEATURES")

    if features_df.shape != (
        EXPECTED_NODES,
        NUM_FEATURES + 2
    ):
        raise ValueError(
            f"Unexpected features shape: {features_df.shape}"
        )

    # -----------------------------------------------------
    # Transaction IDs
    # -----------------------------------------------------

    tx_ids = convert_to_int64(
        features_df.iloc[:, 0],
        "Features txId"
    ).to_numpy(dtype=np.int64)

    if len(np.unique(tx_ids)) != len(tx_ids):
        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    print(f"Number of transactions: {len(tx_ids):,}")

    # -----------------------------------------------------
    # Timestep
    # -----------------------------------------------------

    timestep = convert_to_int64(
        features_df.iloc[:, 1],
        "Timestep"
    ).to_numpy(dtype=np.int8)

    unique_timesteps = np.unique(timestep)

    expected_timesteps = np.arange(1, 50)

    if not np.array_equal(
        unique_timesteps,
        expected_timesteps
    ):
        raise ValueError(
            "Timesteps must range continuously from 1 to 49."
        )

    print(f"Number of timesteps: {len(unique_timesteps)}")

    # -----------------------------------------------------
    # Extract features
    # -----------------------------------------------------

    # Columns 2-94: 93 local features
    X_local = features_df.iloc[:, 2:95].to_numpy(
        dtype=np.float32
    )

    # Columns 95-166: 72 aggregated features
    X_aggregated = features_df.iloc[:, 95:167].to_numpy(
        dtype=np.float32
    )

    # All 165 features
    X_full = np.concatenate(
        [X_local, X_aggregated],
        axis=1
    )

    # Validate dimensions
    assert X_local.shape == (
        EXPECTED_NODES,
        NUM_LOCAL_FEATURES
    )

    assert X_aggregated.shape == (
        EXPECTED_NODES,
        NUM_AGGREGATED_FEATURES
    )

    assert X_full.shape == (
        EXPECTED_NODES,
        NUM_FEATURES
    )

    # Validate numerical values
    if not np.isfinite(X_full).all():
        raise ValueError(
            "Feature matrix contains NaN or Inf."
        )

    print("\nFeature shapes:")
    print("X_local      :", X_local.shape)
    print("X_aggregated :", X_aggregated.shape)
    print("X_full       :", X_full.shape)

    return {
        "tx_ids": tx_ids,
        "timestep": timestep,
        "X_local": X_local,
        "X_aggregated": X_aggregated,
        "X_full": X_full
    }


# =========================================================
# 5. PREPROCESS LABELS
# =========================================================

def preprocess_labels(classes_df, tx_ids):
    """
    Map original labels:

        1       -> 1 (Illicit)
        2       -> 0 (Licit)
        unknown -> -1 (Unknown)

    Align labels with the exact transaction order in features.
    """

    print_section("3. PREPROCESS LABELS")

    required_columns = {"txId", "class"}

    if not required_columns.issubset(classes_df.columns):
        raise ValueError(
            f"Classes must contain columns: {required_columns}"
        )

    # Normalize class transaction IDs
    class_tx_ids = convert_to_int64(
        classes_df["txId"],
        "Classes txId"
    ).to_numpy(dtype=np.int64)

    # Check duplicate IDs
    if len(np.unique(class_tx_ids)) != len(class_tx_ids):
        raise ValueError(
            "Duplicate transaction IDs in classes."
        )

    # Normalize original labels
    original_labels = (
        classes_df["class"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Convert labels
    mapped_labels = original_labels.map(LABEL_MAPPING)

    if mapped_labels.isna().any():
        invalid_labels = original_labels[
            mapped_labels.isna()
        ].unique()

        raise ValueError(
            f"Unknown labels detected: {invalid_labels}"
        )

    mapped_labels = mapped_labels.to_numpy(
        dtype=np.int8
    )

    # -----------------------------------------------------
    # Align labels by transaction ID
    # -----------------------------------------------------

    label_by_txid = pd.Series(
        mapped_labels,
        index=class_tx_ids
    )

    y_series = label_by_txid.reindex(tx_ids)

    if y_series.isna().any():
        raise ValueError(
            "Some feature transaction IDs have no labels."
        )

    y = y_series.to_numpy(dtype=np.int8)

    # -----------------------------------------------------
    # Print distribution
    # -----------------------------------------------------

    illicit_count = int((y == 1).sum())
    licit_count = int((y == 0).sum())
    unknown_count = int((y == -1).sum())

    print("\nMapped label distribution:")
    print(f"Illicit : {illicit_count:,}")
    print(f"Licit   : {licit_count:,}")
    print(f"Unknown : {unknown_count:,}")

    print("\nLabel mapping:")
    print("1  -> Illicit")
    print("0  -> Licit")
    print("-1 -> Unknown")

    if not np.isin(y, [-1, 0, 1]).all():
        raise ValueError(
            "Invalid numerical labels."
        )

    return y


# =========================================================
# 6. MAP TRANSACTION IDs TO NODE INDICES
# =========================================================

def build_node_mapping(tx_ids):
    """
    Create a mapping:

        txId -> node_idx

    node_idx follows the original feature CSV row order.
    """

    print_section("4. BUILD NODE MAPPING")

    num_nodes = len(tx_ids)

    node_indices = np.arange(
        num_nodes,
        dtype=np.int64
    )

    txid_to_idx = pd.Series(
        node_indices,
        index=tx_ids
    )

    print(f"Number of nodes: {num_nodes:,}")
    print(f"Node index range: 0 - {num_nodes - 1}")

    # Example mapping
    print("\nFirst 5 mappings:")

    for i in range(min(5, num_nodes)):
        print(f"{tx_ids[i]} -> {node_indices[i]}")

    return txid_to_idx


# =========================================================
# 7. MAP EDGES
# =========================================================

def build_edge_index(edges_df, txid_to_idx, timestep):
    """
    Convert original transaction IDs into node indices.

    Original edge:
        txId1 -> txId2

    Mapped edge:
        node_idx_1 -> node_idx_2

    Output:
        edge_index with shape (2, E)
    """

    print_section("5. BUILD EDGE INDEX")

    required_columns = {"txId1", "txId2"}

    if not required_columns.issubset(edges_df.columns):
        raise ValueError(
            f"Edges must contain columns: {required_columns}"
        )

    source_ids = convert_to_int64(
        edges_df["txId1"],
        "Edge source IDs"
    )

    target_ids = convert_to_int64(
        edges_df["txId2"],
        "Edge target IDs"
    )

    # -----------------------------------------------------
    # Map original IDs to node indices
    # -----------------------------------------------------

    source_indices = source_ids.map(txid_to_idx)
    target_indices = target_ids.map(txid_to_idx)

    if source_indices.isna().any():
        raise ValueError(
            "Some source IDs do not exist in features."
        )

    if target_indices.isna().any():
        raise ValueError(
            "Some target IDs do not exist in features."
        )

    source_indices = source_indices.to_numpy(
        dtype=np.int64
    )

    target_indices = target_indices.to_numpy(
        dtype=np.int64
    )

    # -----------------------------------------------------
    # Build edge_index
    # -----------------------------------------------------

    edge_index = np.stack(
        [source_indices, target_indices],
        axis=0
    )

    # -----------------------------------------------------
    # Validate edge_index
    # -----------------------------------------------------

    if edge_index.shape != (
        2,
        len(edges_df)
    ):
        raise ValueError(
            f"Invalid edge_index shape: {edge_index.shape}"
        )

    num_nodes = len(timestep)

    if edge_index.min() < 0:
        raise ValueError(
            "Negative node index detected."
        )

    if edge_index.max() >= num_nodes:
        raise ValueError(
            "Node index exceeds the number of nodes."
        )

    # -----------------------------------------------------
    # Check temporal consistency
    # -----------------------------------------------------

    source_timestep = timestep[edge_index[0]]
    target_timestep = timestep[edge_index[1]]

    cross_timestep_edges = (
        source_timestep != target_timestep
    ).sum()

    if cross_timestep_edges > 0:
        raise ValueError(
            f"Found {cross_timestep_edges} cross-timestep edges."
        )

    print(f"Number of edges: {edge_index.shape[1]:,}")
    print(f"edge_index shape: {edge_index.shape}")
    print(f"Cross-timestep edges: {cross_timestep_edges}")

    print("\nFirst 5 mapped edges:")

    for i in range(min(5, edge_index.shape[1])):
        src = edge_index[0, i]
        dst = edge_index[1, i]

        print(f"{src} -> {dst}")

    return edge_index


# =========================================================
# 8. CREATE TEMPORAL MASKS
# =========================================================

def create_temporal_masks(timestep, y):
    """
    Create temporal masks.

    Time masks:
        Include all nodes, including unknown nodes.

    Label masks:
        Include only labeled nodes.

    Split:
        Train: timestep 1-28
        Val  : timestep 29-34
        Test : timestep 35-49
    """

    print_section("6. CREATE TEMPORAL MASKS")

    # -----------------------------------------------------
    # All-node temporal masks
    # -----------------------------------------------------

    train_time_mask = np.isin(
        timestep,
        TRAIN_TIMESTEPS
    )

    val_time_mask = np.isin(
        timestep,
        VAL_TIMESTEPS
    )

    test_time_mask = np.isin(
        timestep,
        TEST_TIMESTEPS
    )

    # -----------------------------------------------------
    # Label masks
    # -----------------------------------------------------

    labeled_mask = y >= 0
    unknown_mask = y == -1

    # -----------------------------------------------------
    # Supervised learning masks
    # -----------------------------------------------------

    train_mask = (
        train_time_mask
        & labeled_mask
    )

    val_mask = (
        val_time_mask
        & labeled_mask
    )

    test_mask = (
        test_time_mask
        & labeled_mask
    )

    # All nodes are retained for graph processing
    all_node_mask = np.ones(
        len(y),
        dtype=bool
    )

    masks = {
        "train_time_mask": train_time_mask,
        "val_time_mask": val_time_mask,
        "test_time_mask": test_time_mask,

        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,

        "labeled_mask": labeled_mask,
        "unknown_mask": unknown_mask,
        "all_node_mask": all_node_mask
    }

    # -----------------------------------------------------
    # Print counts
    # -----------------------------------------------------

    print("\nAll nodes by temporal split:")

    print(
        f"Train timesteps (1-28): "
        f"{train_time_mask.sum():,}"
    )

    print(
        f"Validation timesteps (29-34): "
        f"{val_time_mask.sum():,}"
    )

    print(
        f"Test timesteps (35-49): "
        f"{test_time_mask.sum():,}"
    )

    print("\nLabeled nodes by split:")

    print(f"Train: {train_mask.sum():,}")
    print(f"Val  : {val_mask.sum():,}")
    print(f"Test : {test_mask.sum():,}")

    print("\nOther masks:")

    print(f"Labeled nodes: {labeled_mask.sum():,}")
    print(f"Unknown nodes: {unknown_mask.sum():,}")
    print(f"All nodes    : {all_node_mask.sum():,}")

    # -----------------------------------------------------
    # Validate masks
    # -----------------------------------------------------

    time_overlap = (
        train_time_mask.astype(np.int8)
        + val_time_mask.astype(np.int8)
        + test_time_mask.astype(np.int8)
    )

    if not np.all(time_overlap == 1):
        raise ValueError(
            "Temporal masks overlap or do not cover all nodes."
        )

    supervised_overlap = (
        train_mask.astype(np.int8)
        + val_mask.astype(np.int8)
        + test_mask.astype(np.int8)
    )

    if not np.all(supervised_overlap <= 1):
        raise ValueError(
            "Train/validation/test masks overlap."
        )

    supervised_union = (
        train_mask
        | val_mask
        | test_mask
    )

    if not np.array_equal(
        supervised_union,
        labeled_mask
    ):
        raise ValueError(
            "Some labeled nodes are not assigned to any split."
        )

    if np.any(
        (train_mask | val_mask | test_mask)
        & unknown_mask
    ):
        raise ValueError(
            "Unknown nodes must not be included in supervised masks."
        )

    print("\n[PASS] Temporal masks are valid.")
    print("[PASS] No overlap between supervised splits.")
    print("[PASS] Unknown nodes excluded from supervised masks.")

    return masks


# =========================================================
# 9. BUILD PROCESSED DATASET
# =========================================================

def build_dataset():
    """
    Execute the complete preprocessing pipeline.

    Returns a dictionary containing:
        - Feature matrices
        - Labels
        - Transaction IDs
        - Timesteps
        - Edge index
        - Temporal masks
    """

    print_section("ELLIPTIC DATA PREPROCESSING")

    # 1. Load raw files
    features_df, classes_df, edges_df = load_raw_data()

    # 2. Preprocess features
    feature_data = preprocess_features(features_df)

    tx_ids = feature_data["tx_ids"]
    timestep = feature_data["timestep"]

    X_local = feature_data["X_local"]
    X_aggregated = feature_data["X_aggregated"]
    X_full = feature_data["X_full"]

    # 3. Preprocess labels
    y = preprocess_labels(
        classes_df,
        tx_ids
    )

    # 4. Build mapping
    txid_to_idx = build_node_mapping(tx_ids)

    # 5. Convert edges
    edge_index = build_edge_index(
        edges_df,
        txid_to_idx,
        timestep
    )

    # 6. Create temporal masks
    masks = create_temporal_masks(
        timestep,
        y
    )

    # -----------------------------------------------------
    # Final dataset
    # -----------------------------------------------------

    data = {
        "tx_ids": tx_ids,
        "timestep": timestep,

        "X_local": X_local,
        "X_aggregated": X_aggregated,
        "X_full": X_full,

        "y": y,
        "edge_index": edge_index,

        **masks
    }

    return data


# =========================================================
# 10. SAVE PROCESSED DATASET
# =========================================================

def save_dataset(data, output_path=OUTPUT_PATH):
    """
    Save processed data in compressed NumPy format.

    npz allows P1, P2, P3 and P4 to load the same dataset
    without requiring PyTorch.
    """

    print_section("7. SAVE PROCESSED DATASET")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_path,
        **data
    )

    file_size_mb = (
        output_path.stat().st_size
        / (1024 ** 2)
    )

    print(f"Saved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")


# =========================================================
# 11. VALIDATE PROCESSED DATASET
# =========================================================

def validate_dataset(data):
    """
    Perform final consistency checks before saving.
    """

    print_section("8. VALIDATE PROCESSED DATASET")

    num_nodes = len(data["tx_ids"])
    num_edges = data["edge_index"].shape[1]

    # -----------------------------------------------------
    # Shape checks
    # -----------------------------------------------------

    expected_shapes = {
        "X_local": (num_nodes, 93),
        "X_aggregated": (num_nodes, 72),
        "X_full": (num_nodes, 165),

        "y": (num_nodes,),
        "timestep": (num_nodes,),

        "train_mask": (num_nodes,),
        "val_mask": (num_nodes,),
        "test_mask": (num_nodes,)
    }

    for key, expected_shape in expected_shapes.items():

        actual_shape = data[key].shape

        if actual_shape != expected_shape:
            raise ValueError(
                f"{key}: expected {expected_shape}, "
                f"got {actual_shape}"
            )

        print(f"[PASS] {key}: {actual_shape}")

    # -----------------------------------------------------
    # Edge checks
    # -----------------------------------------------------

    assert data["edge_index"].shape == (
        2,
        EXPECTED_EDGES
    )

    assert num_edges == EXPECTED_EDGES
    assert num_nodes == EXPECTED_NODES

    # -----------------------------------------------------
    # Feature checks
    # -----------------------------------------------------

    assert np.isfinite(data["X_full"]).all()

    assert np.array_equal(
        data["X_full"],
        np.concatenate(
            [
                data["X_local"],
                data["X_aggregated"]
            ],
            axis=1
        )
    )

    # -----------------------------------------------------
    # Label checks
    # -----------------------------------------------------

    assert np.all(
        np.isin(data["y"], [-1, 0, 1])
    )

    # -----------------------------------------------------
    # Mask checks
    # -----------------------------------------------------

    assert not np.any(
        data["train_mask"] & data["val_mask"]
    )

    assert not np.any(
        data["train_mask"] & data["test_mask"]
    )

    assert not np.any(
        data["val_mask"] & data["test_mask"]
    )

    assert np.array_equal(
        data["train_mask"]
        | data["val_mask"]
        | data["test_mask"],
        data["labeled_mask"]
    )

    print("\n[PASS] Feature matrices are consistent.")
    print("[PASS] Labels are valid.")
    print("[PASS] Temporal masks are valid.")
    print("[PASS] Dataset dimensions are valid.")

    print("\nAll processed data checks passed!")


# =========================================================
# 12. MAIN
# =========================================================

def main():

    # Build dataset
    data = build_dataset()

    # Validate dataset
    validate_dataset(data)

    # Save dataset
    save_dataset(data)

    print_section("PREPROCESSING COMPLETED")

    print("Output file:")
    print(OUTPUT_PATH)

    print("\nDataset keys:")

    for key, value in data.items():

        print(
            f"{key:20s} "
            f"shape={str(value.shape):20s} "
            f"dtype={value.dtype}"
        )


if __name__ == "__main__":
    main()

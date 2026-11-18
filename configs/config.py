from pathlib import Path

# =====================================================
# Project paths
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = ROOT_DIR / "outputs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


# =====================================================
# Random seed
# =====================================================

SEED = 42


# =====================================================
# Temporal split
# =====================================================

TRAIN_TIMESTEPS = list(range(1, 29))
VAL_TIMESTEPS = list(range(29, 35))
TEST_TIMESTEPS = list(range(35, 50))


# =====================================================
# Dataset information
# =====================================================

NUM_NODES = 203769
NUM_EDGES = 234355
NUM_FEATURES = 165

NUM_CLASSES = 2
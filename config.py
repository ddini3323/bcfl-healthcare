"""
Central configuration for the BCFL Healthcare project.

The NIH ChestX-ray14 dataset is NOT downloaded automatically.

Dataset location priority:
1. CXR_DATA_DIR environment variable
2. Local ./data/nih_chest_xrays directory

Dataset source:
https://www.kaggle.com/datasets/nih-chest-xrays/data
"""

import os


# ----------------------------------------------------------------------
# Dataset configuration
# ----------------------------------------------------------------------

def _resolve_data_dir():
    """
    Resolve the NIH ChestX-ray14 dataset location.

    Priority:
    1. CXR_DATA_DIR environment variable
    2. Local ./data/nih_chest_xrays directory

    IMPORTANT:
    The dataset is NOT downloaded automatically.
    Use download_dataset.py when you are ready to download it.
    """

    # 1. Explicit environment variable
    env_path = os.environ.get("CXR_DATA_DIR")

    if env_path:
        return os.path.abspath(env_path)

    # 2. Local dataset directory
    local_default = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "nih_chest_xrays"
    )

    return local_default


# ----------------------------------------------------------------------
# Dataset paths
# ----------------------------------------------------------------------

DATA_DIR = _resolve_data_dir()

CSV_PATH = os.path.join(
    DATA_DIR,
    "Data_Entry_2017.csv"
)

TRAIN_VAL_LIST = os.path.join(
    DATA_DIR,
    "train_val_list.txt"
)

TEST_LIST = os.path.join(
    DATA_DIR,
    "test_list.txt"
)

# NIH ChestX-ray14 stores images under DATA_DIR/images_XXX/
IMAGES_DIR = DATA_DIR


# ----------------------------------------------------------------------
# Image / data settings
# ----------------------------------------------------------------------

IMAGE_SIZE = (224, 224)

CHANNELS = 3

INPUT_SHAPE = IMAGE_SIZE + (CHANNELS,)

BATCH_SIZE = 32


# ----------------------------------------------------------------------
# Federated Learning settings
# ----------------------------------------------------------------------

NUM_CLIENTS = 3

NUM_ROUNDS = 5

LOCAL_EPOCHS = 1

LEARNING_RATE = 1e-3

# Proximal term weight.
# 0 disables FedProx and gives plain FedAvg.
FEDPROX_MU = 0.01

RANDOM_SEED = 42


# ----------------------------------------------------------------------
# Privacy settings
# ----------------------------------------------------------------------

# L2 clipping norm applied to each client's weight update
DP_CLIP_NORM = 1.0

# Gaussian noise multiplier
# Noise standard deviation =
# DP_NOISE_MULTIPLIER * DP_CLIP_NORM
DP_NOISE_MULTIPLIER = 0.3

# Target delta for the (epsilon, delta)-DP accountant
DP_DELTA = 1e-5

# Enable secure aggregation
USE_SECURE_AGGREGATION = True


# ----------------------------------------------------------------------
# Output locations
# ----------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs"
)

BLOCKCHAIN_LOG = os.path.join(
    OUTPUT_DIR,
    "blockchain_ledger.json"
)

METRICS_LOG = os.path.join(
    OUTPUT_DIR,
    "metrics_history.json"
)


# ----------------------------------------------------------------------
# Create output directories if they don't exist
# ----------------------------------------------------------------------

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Configuration summary
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("BCFL Healthcare Configuration")
    print("=" * 60)

    print(f"Dataset directory : {DATA_DIR}")
    print(f"CSV path          : {CSV_PATH}")
    print(f"Train/Val list    : {TRAIN_VAL_LIST}")
    print(f"Test list         : {TEST_LIST}")
    print(f"Images directory  : {IMAGES_DIR}")

    print("-" * 60)

    print(f"Image size        : {IMAGE_SIZE}")
    print(f"Batch size        : {BATCH_SIZE}")

    print("-" * 60)

    print(f"Number of clients : {NUM_CLIENTS}")
    print(f"Federated rounds  : {NUM_ROUNDS}")
    print(f"Local epochs      : {LOCAL_EPOCHS}")
    print(f"Learning rate     : {LEARNING_RATE}")
    print(f"FedProx mu        : {FEDPROX_MU}")

    print("-" * 60)

    print(f"DP clip norm      : {DP_CLIP_NORM}")
    print(f"DP noise          : {DP_NOISE_MULTIPLIER}")
    print(f"DP delta          : {DP_DELTA}")
    print(f"Secure aggregation: {USE_SECURE_AGGREGATION}")

    print("-" * 60)

    print(f"Checkpoint dir    : {CHECKPOINT_DIR}")
    print(f"Output dir        : {OUTPUT_DIR}")
    print(f"Blockchain log    : {BLOCKCHAIN_LOG}")
    print(f"Metrics log       : {METRICS_LOG}")

    print("=" * 60)
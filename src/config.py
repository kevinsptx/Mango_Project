import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = PROJECT_ROOT / "mango"

DATA_DIR = Path(
    os.environ.get(
        "MANGO_DATA_DIR",
        str(DEFAULT_DATA_DIR)
    )
)

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = DATA_PROCESSED_DIR / "manifest.csv"

SPLIT_MANIFEST_PATH = DATA_PROCESSED_DIR / "manifest_split.csv"
MODELS_DIR = PROJECT_ROOT / "models"


CLASSES = [
    "OverRipe",
    "Ripe",
    "UnRipe"
]


IMG_SIZE = (128, 128)


IMG_CHANNELS = 3


BATCH_SIZE = 8

SEED = 42

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}
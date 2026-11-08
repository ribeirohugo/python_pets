"""Project paths, labels and environment settings."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"  # original images
PROCESSED_DIR = DATASETS_DIR / "processed"  # derived data (nothing writes here yet)
ANNOTATIONS_DIR = DATASETS_DIR / "annotations"  # Pascal-VOC .xml labels
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "catdog_detector.pt"
STATIC_DIR = Path(__file__).parent / "static"

# Index 0 is always "background" for torchvision detectors.
CLASSES = ["background", "cat", "dog"]

# COCO category ids for cat / dog in the pretrained model (used as a fallback
# before you have trained your own weights).
COCO_TO_OURS = {17: 1, 18: 2}


def dataset_url() -> str:
    """Return DATASET_URL from the environment or .env, or "" if it isn't set."""
    load_dotenv(PROJECT_ROOT / ".env")
    return os.environ.get("DATASET_URL", "").strip()


def server_port() -> int:
    """Return PORT from the environment or .env, or 8010 if it isn't set."""
    load_dotenv(PROJECT_ROOT / ".env")
    return int(os.environ.get("PORT", "").strip() or 8010)

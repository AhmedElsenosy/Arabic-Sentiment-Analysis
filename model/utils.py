"""
Shared utilities for the Arabic Sentiment Analysis project.

Handles:
- Logging configuration
- Random seed setting for reproducibility
- Device detection (GPU/CPU)
- Configuration loading (placeholder for now)
"""

import os
import random
import torch
import numpy as np
import logging
from pathlib import Path


def setup_logging(log_file: str = "app.log") -> logging.Logger:
    """Configure and return a standard logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("sentiment_analysis")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across torch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may slightly impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU detected. Running on CPU.")
    return device


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


logger = setup_logging()

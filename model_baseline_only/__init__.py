"""
Baseline-only image captioning model package.
"""

from .model import BaselineCaptionModel
from .dataset import get_data_loaders, ImageCaptionDataset
from .trainer import Trainer, create_default_config

__all__ = [
    "BaselineCaptionModel",
    "ImageCaptionDataset",
    "get_data_loaders",
    "Trainer",
    "create_default_config",
]

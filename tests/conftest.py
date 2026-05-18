"""
Shared pytest fixtures.

All fixtures use synthetic in-memory data — no MNIST download is
performed by the test suite.  The base_cfg fixture is the canonical
minimal config dict used across all test modules.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture()
def base_cfg() -> dict:
    """Minimal valid config dict for all pipeline tests."""
    return {
        "seed": 42,
        "data": {
            "root": "./data",
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": False,
            "val_split": 0.1,
            "augmentation": False,
            "normalize": {"mean": 0.1307, "std": 0.3081},
        },
        "model": {
            "arch": "cnn",
            "conv_channels": [32, 64],
            "fc_hidden": 256,
            "use_batchnorm": True,
            "input_size": 784,
            "hidden_layers": [256, 128],
            "num_classes": 10,
            "activation": "relu",
            "dropout": 0.5,
        },
        "training": {
            "epochs": 2,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "momentum": 0.9,
            "lr_scheduler": {"type": "none"},
            "early_stopping": {
                "enabled": False,
                "monitor": "val_loss",
                "patience": 5,
                "min_delta": 0.0,
            },
            "checkpoint": {
                "dir": "./checkpoints",
                "save_best_only": True,
                "save_every_epoch": False,
            },
            "log_dir": "./runs",
            "auto_visualize": False,
        },
        "inference": {
            "checkpoint": "./checkpoints/best.pt",
            "device": "cpu",
        },
    }


@pytest.fixture()
def synthetic_loader() -> DataLoader:
    """
    A DataLoader of 200 synthetic (image, label) pairs shaped (1, 28, 28).
    Useful for testing trainer and predictor without real MNIST.
    """
    images = torch.randn(200, 1, 28, 28)
    labels = torch.randint(0, 10, (200,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=32, shuffle=False)

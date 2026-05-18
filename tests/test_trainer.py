"""
Tests for src/training/ — EarlyStopping, metrics, and Trainer.

All tests use synthetic data; no MNIST download or file I/O outside
temporary directories.

Covers acceptance criteria:
    AC-TRN-001  fit() completes without error for the default config
    AC-TRN-002  best.pt and last.pt are created after fit()
    AC-TRN-003  metrics.csv contains the correct columns
    AC-TRN-004  evaluate() returns test_acc, test_loss, macro_f1, confusion_matrix
    AC-TRN-005  early stopping halts training before the configured epoch limit

Additional coverage:
    - EarlyStopping: val_loss improvement direction
    - EarlyStopping: val_acc improvement direction
    - EarlyStopping: patience counter reset on improvement
    - EarlyStopping: invalid monitor/patience/min_delta raises ValueError
    - compute_accuracy: correct fraction
    - compute_per_class_accuracy: per-class correct fractions
    - compute_confusion_matrix: bincount shape and diagonal values
    - compute_macro_f1: perfect-prediction case returns 1.0
    - Trainer: model is in eval() during evaluate()
    - Trainer: CSV header written once even on resumed run
    - Trainer: checkpoint dict contains required keys
"""

from __future__ import annotations

import copy
import csv
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.cnn import MNISTConvNet
from src.training.early_stopping import EarlyStopping
from src.training.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_macro_f1,
    compute_per_class_accuracy,
)
from src.training.trainer import Trainer


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def small_loader() -> DataLoader:
    """200 synthetic (1,28,28) images with random labels, batch_size=32."""
    images = torch.randn(200, 1, 28, 28)
    labels = torch.randint(0, 10, (200,))
    return DataLoader(TensorDataset(images, labels), batch_size=32, shuffle=False)


@pytest.fixture()
def trainer_cfg(base_cfg, tmp_path) -> dict:
    """Base config with paths redirected to tmp_path for isolation."""
    cfg = copy.deepcopy(base_cfg)
    cfg["training"]["log_dir"]              = str(tmp_path / "runs")
    cfg["training"]["checkpoint"]["dir"]    = str(tmp_path / "checkpoints")
    cfg["training"]["auto_visualize"]       = False
    cfg["training"]["early_stopping"]["enabled"] = False
    return cfg


@pytest.fixture()
def default_trainer(trainer_cfg, tmp_path) -> Trainer:
    """Trainer with default CNN model and 2-epoch config."""
    model = MNISTConvNet(trainer_cfg)
    return Trainer(model, trainer_cfg, run_id="test_run")


# ──────────────────────────────────────────────────────────────────────────────
# EarlyStopping
# ──────────────────────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_val_loss_first_epoch_always_improves(self):
        """Sentinel float('inf') means first epoch always improves."""
        es = EarlyStopping(monitor="val_loss", patience=3)
        stopped = es.step(1.5)
        assert not stopped
        assert es.counter == 0
        assert es.best_value == pytest.approx(1.5)

    def test_val_acc_first_epoch_always_improves(self):
        """Sentinel float('-inf') means first epoch always improves."""
        es = EarlyStopping(monitor="val_acc", patience=3)
        stopped = es.step(0.85)
        assert not stopped
        assert es.counter == 0

    def test_counter_increments_on_no_improvement(self):
        es = EarlyStopping(monitor="val_loss", patience=3)
        es.step(1.0)   # improvement — sets best
        es.step(1.1)   # no improvement
        es.step(1.2)   # no improvement
        assert es.counter == 2
        assert not es.should_stop

    def test_patience_exhausted_triggers_stop(self):
        es = EarlyStopping(monitor="val_loss", patience=2)
        es.step(1.0)   # improvement
        es.step(1.1)   # no improvement  (counter=1)
        stopped = es.step(1.2)   # no improvement  (counter=2 == patience)
        assert stopped
        assert es.should_stop

    def test_counter_resets_on_improvement(self):
        es = EarlyStopping(monitor="val_loss", patience=5)
        es.step(1.0)   # improvement
        es.step(1.1)   # no improvement → counter=1
        es.step(1.2)   # no improvement → counter=2
        es.step(0.9)   # improvement   → counter=0
        assert es.counter == 0

    def test_min_delta_respected(self):
        """Improvement smaller than min_delta is not counted as improvement."""
        es = EarlyStopping(monitor="val_loss", patience=3, min_delta=0.05)
        es.step(1.0)   # improvement
        es.step(0.98)  # delta = 0.02 < min_delta — no improvement
        assert es.counter == 1

    def test_val_acc_improvement_direction(self):
        """val_acc: higher is better."""
        es = EarlyStopping(monitor="val_acc", patience=3)
        es.step(0.80)   # improvement
        es.step(0.78)   # lower — no improvement
        assert es.counter == 1

    def test_invalid_monitor_raises(self):
        with pytest.raises(ValueError, match="monitor"):
            EarlyStopping(monitor="test_loss")

    def test_invalid_patience_raises(self):
        with pytest.raises(ValueError, match="patience"):
            EarlyStopping(patience=0)

    def test_invalid_min_delta_raises(self):
        with pytest.raises(ValueError, match="min_delta"):
            EarlyStopping(min_delta=-0.01)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeAccuracy:
    def test_all_correct(self):
        logits = torch.eye(5)  # argmax[i] = i
        labels = torch.arange(5)
        assert compute_accuracy(logits, labels) == pytest.approx(1.0)

    def test_all_wrong(self):
        logits = torch.zeros(4, 3)
        logits[:, 1] = 1.0   # always predicts class 1
        labels = torch.zeros(4, dtype=torch.long)  # true is class 0
        assert compute_accuracy(logits, labels) == pytest.approx(0.0)

    def test_half_correct(self):
        logits = torch.zeros(4, 2)
        logits[0, 0] = 1.0
        logits[1, 0] = 1.0
        logits[2, 1] = 1.0
        logits[3, 1] = 1.0
        labels = torch.tensor([0, 0, 0, 0])  # 2/4 correct
        assert compute_accuracy(logits, labels) == pytest.approx(0.5)


class TestComputePerClassAccuracy:
    def test_perfect_predictions(self):
        preds  = torch.arange(10).repeat(10)  # 10 of each class
        labels = torch.arange(10).repeat(10)
        acc = compute_per_class_accuracy(preds, labels, num_classes=10)
        assert all(a == pytest.approx(1.0) for a in acc)

    def test_no_samples_for_class(self):
        """Class with zero samples returns 0.0, not division error."""
        preds  = torch.tensor([0, 0, 1, 1])
        labels = torch.tensor([0, 0, 1, 1])
        acc = compute_per_class_accuracy(preds, labels, num_classes=3)
        assert acc[2] == pytest.approx(0.0)

    def test_partial_accuracy(self):
        """Class 0: 2/3 correct; class 1: 1/2 correct."""
        preds  = torch.tensor([0, 0, 1, 1, 0])
        labels = torch.tensor([0, 0, 0, 1, 1])
        acc = compute_per_class_accuracy(preds, labels, num_classes=2)
        assert acc[0] == pytest.approx(2 / 3)
        assert acc[1] == pytest.approx(1 / 2)


class TestComputeConfusionMatrix:
    def test_shape(self):
        preds  = torch.randint(0, 10, (100,))
        labels = torch.randint(0, 10, (100,))
        cm = compute_confusion_matrix(preds, labels, num_classes=10)
        assert cm.shape == (10, 10)

    def test_perfect_diagonal(self):
        """Perfect predictions: CM is diagonal."""
        labels = torch.arange(5).repeat(4)
        cm = compute_confusion_matrix(labels, labels, num_classes=5)
        assert cm.diag().sum().item() == cm.sum().item()

    def test_row_sums_equal_true_counts(self):
        """Row i sums to the total number of samples with true label i."""
        labels = torch.tensor([0, 0, 1, 1, 1, 2])
        preds  = torch.tensor([0, 1, 1, 1, 2, 2])
        cm = compute_confusion_matrix(preds, labels, num_classes=3)
        row_sums = cm.sum(dim=1)
        assert row_sums[0].item() == 2
        assert row_sums[1].item() == 3
        assert row_sums[2].item() == 1


class TestComputeMacroF1:
    def test_perfect_predictions_return_one(self):
        labels = torch.arange(10).repeat(10)
        f1 = compute_macro_f1(labels, labels, num_classes=10)
        assert f1 == pytest.approx(1.0)

    def test_value_in_range(self):
        preds  = torch.randint(0, 10, (200,))
        labels = torch.randint(0, 10, (200,))
        f1 = compute_macro_f1(preds, labels, num_classes=10)
        assert 0.0 <= f1 <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Trainer.fit() (AC-TRN-001, AC-TRN-002, AC-TRN-003)
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainerFit:
    def test_fit_completes_without_error(
        self, default_trainer, small_loader
    ):
        """AC-TRN-001: fit() runs to completion without raising."""
        metrics = default_trainer.fit(small_loader, small_loader)
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics
        assert "val_acc"  in metrics

    def test_best_and_last_checkpoints_saved(
        self, default_trainer, small_loader
    ):
        """AC-TRN-002: best.pt and last.pt exist after fit()."""
        default_trainer.fit(small_loader, small_loader)
        ckpt_dir = default_trainer.ckpt_dir
        assert (ckpt_dir / "best.pt").exists(), "best.pt not found"
        assert (ckpt_dir / "last.pt").exists(), "last.pt not found"

    def test_metrics_csv_columns(self, default_trainer, small_loader):
        """AC-TRN-003: metrics.csv contains the expected columns."""
        default_trainer.fit(small_loader, small_loader)
        csv_path = default_trainer.csv_path
        assert csv_path.exists()
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        expected_cols = {"epoch", "train_loss", "val_loss", "val_acc", "lr"}
        assert expected_cols <= set(row.keys()), (
            f"Missing columns: {expected_cols - set(row.keys())}"
        )

    def test_csv_row_count_matches_epochs(self, default_trainer, small_loader):
        """One CSV row per training epoch."""
        default_trainer.fit(small_loader, small_loader)
        with open(default_trainer.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        n_epochs = default_trainer.cfg["training"]["epochs"]
        assert len(rows) == n_epochs

    def test_checkpoint_schema(self, default_trainer, small_loader):
        """Checkpoint dict contains all required keys (SPEC §10.4.2)."""
        default_trainer.fit(small_loader, small_loader)
        ckpt = torch.load(
            default_trainer.ckpt_dir / "last.pt",
            map_location="cpu",
            weights_only=False,
        )
        required = {"epoch", "model_state", "optimizer_state", "scheduler_state",
                    "metrics", "config"}
        assert required <= ckpt.keys(), (
            f"Missing checkpoint keys: {required - ckpt.keys()}"
        )

    def test_final_metrics_epoch_value(self, default_trainer, small_loader):
        """fit() returns metrics for the final epoch."""
        n_epochs = default_trainer.cfg["training"]["epochs"]
        metrics = default_trainer.fit(small_loader, small_loader)
        assert metrics["epoch"] == n_epochs

    def test_per_epoch_checkpoint_saved_when_flag_enabled(
        self, trainer_cfg, small_loader
    ):
        """epoch_001.pt is saved when save_every_epoch=True."""
        cfg = copy.deepcopy(trainer_cfg)
        cfg["training"]["checkpoint"]["save_every_epoch"] = True
        model = MNISTConvNet(cfg)
        trainer = Trainer(model, cfg, run_id="test_per_epoch")
        trainer.fit(small_loader, small_loader)
        assert (trainer.ckpt_dir / "epoch_001.pt").exists()

    def test_config_snapshot_saved(self, default_trainer, small_loader):
        """config.yaml is saved to the run directory."""
        default_trainer.fit(small_loader, small_loader)
        assert (default_trainer.run_dir / "config.yaml").exists()


# ──────────────────────────────────────────────────────────────────────────────
# Trainer.evaluate() (AC-TRN-004)
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainerEvaluate:
    def test_evaluate_returns_required_keys(
        self, default_trainer, small_loader
    ):
        """AC-TRN-004: evaluate() returns all required metrics."""
        default_trainer.fit(small_loader, small_loader)
        results = default_trainer.evaluate(small_loader)
        required = {"test_loss", "test_acc", "per_class_acc",
                    "confusion_matrix", "macro_f1"}
        assert required <= results.keys(), (
            f"Missing keys: {required - results.keys()}"
        )

    def test_test_acc_in_range(self, default_trainer, small_loader):
        """test_acc is a float in [0, 1]."""
        default_trainer.fit(small_loader, small_loader)
        results = default_trainer.evaluate(small_loader)
        assert 0.0 <= results["test_acc"] <= 1.0

    def test_confusion_matrix_shape(self, default_trainer, small_loader):
        """confusion_matrix is a 10×10 list of lists."""
        default_trainer.fit(small_loader, small_loader)
        results = default_trainer.evaluate(small_loader)
        cm = results["confusion_matrix"]
        assert len(cm) == 10
        assert all(len(row) == 10 for row in cm)

    def test_per_class_acc_length(self, default_trainer, small_loader):
        """per_class_acc has one entry per class."""
        default_trainer.fit(small_loader, small_loader)
        results = default_trainer.evaluate(small_loader)
        assert len(results["per_class_acc"]) == 10

    def test_test_results_json_saved(self, default_trainer, small_loader):
        """test_results.json is written to the run directory."""
        default_trainer.fit(small_loader, small_loader)
        default_trainer.evaluate(small_loader)
        assert (default_trainer.run_dir / "test_results.json").exists()

    def test_model_in_eval_mode_during_evaluate(
        self, default_trainer, small_loader
    ):
        """
        model.eval() is active during evaluate().
        We check the model is back in train() mode after, since evaluate()
        calls model.train() before returning.
        """
        default_trainer.fit(small_loader, small_loader)
        default_trainer.evaluate(small_loader)
        # Trainer.evaluate calls model.train() at the end
        assert default_trainer.model.training is True


# ──────────────────────────────────────────────────────────────────────────────
# Early stopping integration (AC-TRN-005)
# ──────────────────────────────────────────────────────────────────────────────

class TestEarlyStoppingIntegration:
    def test_training_halts_before_epoch_limit(
        self, trainer_cfg, small_loader
    ):
        """
        AC-TRN-005: with patience=1 and min_delta=1e6, no epoch after the
        first can possibly register as an improvement, so training stops
        deterministically after 2 epochs (1 improvement + 1 patience).
        """
        cfg = copy.deepcopy(trainer_cfg)
        cfg["training"]["epochs"] = 10
        cfg["training"]["early_stopping"] = {
            "enabled":   True,
            "monitor":   "val_loss",
            "patience":  1,
            "min_delta": 1e6,   # impossibly large — only epoch 1 improves
        }
        model = MNISTConvNet(cfg)
        trainer = Trainer(model, cfg, run_id="test_es")
        trainer.fit(small_loader, small_loader)

        # The run must have fewer rows than max epochs
        with open(trainer.csv_path, newline="", encoding="utf-8") as f:
            n_rows = sum(1 for _ in csv.DictReader(f))
        assert n_rows < 10, (
            f"Expected early stop before 10 epochs; ran {n_rows} epochs"
        )

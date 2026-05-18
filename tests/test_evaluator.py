"""
Tests for src/evaluation/evaluator.py and the extended metrics.

All tests use synthetic tensors or temporary checkpoints; no MNIST download
is performed.

Covers acceptance criteria:
    AC-EVL-001  Evaluator loads best.pt and returns required metrics
    AC-EVL-002  confusion_matrix.png is generated after run()
    AC-EVL-003  test_results.json is generated with all required fields
    AC-EVL-004  per-class precision / recall / F1 are correct for known input

Additional coverage:
    - compute_classification_report: perfect-prediction case
    - compute_classification_report: zero-support class handled gracefully
    - compute_classification_report: macro averages are unweighted means
    - Evaluator raises FileNotFoundError for missing checkpoint
    - build_model returns MNISTConvNet for arch='cnn'
    - build_model raises NotImplementedError for arch='mlp'
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import build_model
from src.training.metrics import compute_classification_report


# ──────────────────────────────────────────────────────────────────────────────
# compute_classification_report
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeClassificationReport:
    def test_perfect_predictions(self):
        """All metrics equal 1.0 when predictions are perfect."""
        labels = torch.arange(10).repeat(100)
        report = compute_classification_report(labels, labels, num_classes=10)

        assert report["accuracy"] == pytest.approx(1.0)
        assert report["macro"]["precision"] == pytest.approx(1.0)
        assert report["macro"]["recall"]    == pytest.approx(1.0)
        assert report["macro"]["f1"]        == pytest.approx(1.0)

        for cls in report["per_class"]:
            assert cls["precision"] == pytest.approx(1.0)
            assert cls["recall"]    == pytest.approx(1.0)
            assert cls["f1"]        == pytest.approx(1.0)

    def test_report_structure(self):
        """Return dict has expected top-level and nested keys."""
        preds  = torch.randint(0, 10, (200,))
        labels = torch.randint(0, 10, (200,))
        report = compute_classification_report(preds, labels)

        assert set(report.keys()) == {"per_class", "macro", "accuracy", "n_samples"}
        assert set(report["macro"].keys()) == {"precision", "recall", "f1"}
        for cls in report["per_class"]:
            assert set(cls.keys()) == {"class", "precision", "recall", "f1", "support"}

    def test_n_samples_matches_input(self):
        preds  = torch.randint(0, 10, (300,))
        labels = torch.randint(0, 10, (300,))
        report = compute_classification_report(preds, labels)
        assert report["n_samples"] == 300

    def test_per_class_length(self):
        preds  = torch.randint(0, 5, (100,))
        labels = torch.randint(0, 5, (100,))
        report = compute_classification_report(preds, labels, num_classes=5)
        assert len(report["per_class"]) == 5

    def test_zero_support_class_no_error(self):
        """A class with no true samples produces 0.0, not ZeroDivisionError."""
        preds  = torch.tensor([0, 0, 1, 1])
        labels = torch.tensor([0, 0, 1, 1])   # class 2 has no samples
        report = compute_classification_report(preds, labels, num_classes=3)
        assert report["per_class"][2]["recall"]    == pytest.approx(0.0)
        assert report["per_class"][2]["precision"] == pytest.approx(0.0)

    def test_precision_recall_known_values(self):
        """
        Hand-computed case:
            preds  = [0, 0, 1, 1, 2]
            labels = [0, 1, 1, 1, 2]
            CM:      [[1,1,0], [0,2,0], [0,0,1]]

        Confusion matrix C[true, pred]:
            preds  = [0, 0, 1, 1, 2]
            labels = [0, 1, 1, 1, 2]
            C = [[1, 0, 0],
                 [1, 2, 0],
                 [0, 0, 1]]

        Class 0:
            TP=1, FP=C[:,0].sum()-TP = 2-1=1  → Precision=0.5
            TP=1, FN=C[0,:].sum()-TP = 1-1=0  → Recall=1.0

        Class 1:
            TP=2, FP=C[:,1].sum()-TP = 2-2=0  → Precision=1.0
            TP=2, FN=C[1,:].sum()-TP = 3-2=1  → Recall=2/3

        Class 2:
            TP=1, FP=0  → Precision=1.0
            TP=1, FN=0  → Recall=1.0
        """
        preds  = torch.tensor([0, 0, 1, 1, 2])
        labels = torch.tensor([0, 1, 1, 1, 2])
        report = compute_classification_report(preds, labels, num_classes=3)

        c = report["per_class"]
        assert c[0]["precision"] == pytest.approx(0.5)
        assert c[0]["recall"]    == pytest.approx(1.0)
        assert c[1]["precision"] == pytest.approx(1.0)
        assert c[1]["recall"]    == pytest.approx(2 / 3, rel=1e-5)
        assert c[2]["precision"] == pytest.approx(1.0)
        assert c[2]["recall"]    == pytest.approx(1.0)

    def test_macro_is_unweighted_mean(self):
        """Macro values equal the unweighted mean of per-class values."""
        preds  = torch.randint(0, 4, (200,))
        labels = torch.randint(0, 4, (200,))
        report = compute_classification_report(preds, labels, num_classes=4)

        expected_macro_p = sum(c["precision"] for c in report["per_class"]) / 4
        assert report["macro"]["precision"] == pytest.approx(expected_macro_p)

    def test_values_in_range(self):
        """All metric values are in [0, 1]."""
        preds  = torch.randint(0, 10, (500,))
        labels = torch.randint(0, 10, (500,))
        report = compute_classification_report(preds, labels)
        for cls in report["per_class"]:
            assert 0.0 <= cls["precision"] <= 1.0
            assert 0.0 <= cls["recall"]    <= 1.0
            assert 0.0 <= cls["f1"]        <= 1.0
        assert 0.0 <= report["accuracy"] <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# build_model factory
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildModel:
    def test_cnn_arch_returns_mnistconvnet(self, base_cfg):
        from src.model.cnn import MNISTConvNet
        model = build_model(base_cfg)
        assert isinstance(model, MNISTConvNet)

    def test_mlp_arch_raises_not_implemented(self, base_cfg):
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["arch"] = "mlp"
        with pytest.raises(NotImplementedError, match="MLP"):
            build_model(cfg)

    def test_unknown_arch_raises_value_error(self, base_cfg):
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["arch"] = "transformer"
        with pytest.raises(ValueError, match="arch"):
            build_model(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator — fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_checkpoint(cfg: dict, tmp_path: Path) -> Path:
    """Write a synthetic best.pt under tmp_path/checkpoints/test_run/."""
    from src.model.cnn import MNISTConvNet
    model = MNISTConvNet(cfg)
    ckpt_dir = tmp_path / "checkpoints" / "test_run"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "best.pt"
    torch.save(
        {
            "epoch":           1,
            "model_state":     model.state_dict(),
            "optimizer_state": {},
            "scheduler_state": None,
            "metrics":         {"val_loss": 0.1, "val_acc": 0.95},
            "config":          cfg,
        },
        ckpt_path,
    )
    return ckpt_path


def _make_test_loader() -> DataLoader:
    images = torch.randn(100, 1, 28, 28)
    labels = torch.randint(0, 10, (100,))
    return DataLoader(TensorDataset(images, labels), batch_size=32)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator tests (AC-EVL-001 … AC-EVL-004)
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluator:
    @pytest.fixture()
    def evaluator(self, base_cfg, tmp_path):
        """Evaluator with a synthetic checkpoint and patched test loader."""
        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["log_dir"] = str(tmp_path / "runs")
        ckpt_path = _make_checkpoint(cfg, tmp_path)

        from src.evaluation.evaluator import Evaluator

        ev = Evaluator(
            checkpoint_path=ckpt_path,
            run_dir=str(tmp_path / "runs" / "test_run"),
        )
        return ev

    def _patch_loader(self, evaluator):
        """Replace _build_test_loader with a synthetic one."""
        evaluator._build_test_loader = _make_test_loader
        return evaluator

    def test_missing_checkpoint_raises(self, tmp_path):
        from src.evaluation.evaluator import Evaluator
        with pytest.raises(FileNotFoundError):
            Evaluator(checkpoint_path=tmp_path / "nonexistent.pt")

    def test_run_returns_required_keys(self, evaluator):
        """AC-EVL-001: run() returns all required metric keys."""
        self._patch_loader(evaluator)
        results = evaluator.run()
        required = {"test_loss", "test_acc", "n_samples", "per_class",
                    "macro", "confusion_matrix"}
        assert required <= results.keys()

    def test_confusion_matrix_png_created(self, evaluator, tmp_path):
        """AC-EVL-002: confusion_matrix.png is written after run()."""
        self._patch_loader(evaluator)
        evaluator.run()
        assert (evaluator.run_dir / "confusion_matrix.png").exists()

    def test_test_results_json_created(self, evaluator):
        """AC-EVL-003: test_results.json is written with required fields."""
        self._patch_loader(evaluator)
        evaluator.run()
        json_path = evaluator.run_dir / "test_results.json"
        assert json_path.exists()
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        required = {"checkpoint", "epoch", "test_loss", "test_acc",
                    "n_samples", "macro", "per_class"}
        assert required <= data.keys()

    def test_per_class_metrics_correct_for_known_input(self, base_cfg, tmp_path):
        """
        AC-EVL-004: evaluator computes correct per-class metrics.

        Use a trivial 'model' that always predicts class 0 on a dataset
        where only class 0 exists, then verify precision=recall=F1=1.0.
        """
        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["log_dir"] = str(tmp_path / "runs")
        ckpt_path = _make_checkpoint(cfg, tmp_path)

        from src.evaluation.evaluator import Evaluator

        ev = Evaluator(
            checkpoint_path=ckpt_path,
            run_dir=str(tmp_path / "runs" / "test_run"),
        )

        # Replace _eval_loop with known output
        labels_t = torch.zeros(50, dtype=torch.long)
        preds_t  = torch.zeros(50, dtype=torch.long)
        from src.training.metrics import compute_classification_report, compute_confusion_matrix
        fake_report = compute_classification_report(preds_t, labels_t, num_classes=10)
        fake_cm     = compute_confusion_matrix(preds_t, labels_t, num_classes=10)

        ev._eval_loop = lambda loader: {
            "test_loss": 0.001,
            "cm":        fake_cm,
            "report":    fake_report,
        }

        results = ev.run()
        assert results["test_acc"] == pytest.approx(1.0)
        # Class 0: all predictions and all labels are 0 → P=R=F1=1
        class0 = next(c for c in results["per_class"] if c["class"] == 0)
        assert class0["precision"] == pytest.approx(1.0)
        assert class0["recall"]    == pytest.approx(1.0)
        assert class0["f1"]        == pytest.approx(1.0)

    def test_macro_values_in_range(self, evaluator):
        """Macro precision, recall, F1 are in [0, 1]."""
        self._patch_loader(evaluator)
        results = evaluator.run()
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= results["macro"][key] <= 1.0

    def test_confusion_matrix_shape(self, evaluator):
        """Returned confusion_matrix is 10×10 (nested list)."""
        self._patch_loader(evaluator)
        results = evaluator.run()
        cm = results["confusion_matrix"]
        assert len(cm) == 10
        assert all(len(row) == 10 for row in cm)

    def test_n_samples_correct(self, evaluator):
        """n_samples equals the total number of test samples evaluated."""
        self._patch_loader(evaluator)
        results = evaluator.run()
        # synthetic loader has 100 images
        assert results["n_samples"] == 100

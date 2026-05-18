"""
Tests for src/utils.py.

All tests are self-contained: no file I/O outside tmp_path, no network,
no MNIST download.

Covers:
    set_seed          — same seed produces identical random sequences
    configure_logging — idempotency, console + file handlers
    AverageMeter      — correct weighted average, n-weighted batches, reset
    save_checkpoint   — writes file; load_checkpoint round-trips it
    load_checkpoint   — raises FileNotFoundError / KeyError on bad input
    plot_training_curves — creates a PNG; empty input is a no-op
    save_json / load_json — round-trip with nested data; sort_keys ordering
    Timer             — context manager and explicit start/stop; elapsed > 0
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils import (
    AverageMeter,
    Timer,
    configure_logging,
    load_checkpoint,
    load_json,
    plot_training_curves,
    save_checkpoint,
    save_json,
    set_seed,
)


# ──────────────────────────────────────────────────────────────────────────────
# set_seed
# ──────────────────────────────────────────────────────────────────────────────

class TestSetSeed:
    def test_same_seed_same_torch_random(self):
        """Two calls with the same seed produce identical PyTorch tensors."""
        set_seed(42)
        t1 = torch.randn(10)
        set_seed(42)
        t2 = torch.randn(10)
        assert torch.equal(t1, t2)

    def test_same_seed_same_python_random(self):
        """Same seed → same Python random sequence."""
        set_seed(7)
        r1 = [random.random() for _ in range(5)]
        set_seed(7)
        r2 = [random.random() for _ in range(5)]
        assert r1 == r2

    def test_same_seed_same_numpy_random(self):
        """Same seed → same NumPy random sequence."""
        set_seed(99)
        a1 = np.random.rand(5).tolist()
        set_seed(99)
        a2 = np.random.rand(5).tolist()
        assert a1 == a2

    def test_different_seeds_different_tensors(self):
        """Different seeds produce different random tensors (with overwhelming probability)."""
        set_seed(1)
        t1 = torch.randn(50)
        set_seed(2)
        t2 = torch.randn(50)
        assert not torch.equal(t1, t2)

    def test_cudnn_flags_set(self):
        """cuDNN deterministic mode is enabled."""
        set_seed(0)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False


# ──────────────────────────────────────────────────────────────────────────────
# configure_logging
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigureLogging:
    def test_returns_root_logger(self):
        root = configure_logging(level="WARNING")
        assert root is logging.getLogger()

    def test_sets_log_level(self):
        configure_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        configure_logging(level="WARNING")   # restore to less verbose

    def test_file_handler_created(self, tmp_path):
        log_file = tmp_path / "test.log"
        configure_logging(level="INFO", log_file=log_file)
        assert log_file.exists()

    def test_idempotent_no_duplicate_file_handlers(self, tmp_path):
        log_file = tmp_path / "dedup.log"
        before = sum(
            1 for h in logging.getLogger().handlers
            if isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == log_file.resolve()
        )
        configure_logging(level="INFO", log_file=log_file)
        configure_logging(level="INFO", log_file=log_file)   # second call
        after = sum(
            1 for h in logging.getLogger().handlers
            if isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).resolve() == log_file.resolve()
        )
        assert after - before == 1, "Handler added more than once"

    def test_parent_dirs_created(self, tmp_path):
        log_file = tmp_path / "nested" / "dir" / "app.log"
        configure_logging(log_file=log_file)
        assert log_file.exists()


# ──────────────────────────────────────────────────────────────────────────────
# AverageMeter
# ──────────────────────────────────────────────────────────────────────────────

class TestAverageMeter:
    def test_initial_avg_is_zero(self):
        """avg returns 0.0 before any update (no division by zero)."""
        m = AverageMeter()
        assert m.avg == pytest.approx(0.0)

    def test_single_update(self):
        m = AverageMeter()
        m.update(2.5, n=4)
        assert m.avg == pytest.approx(2.5)
        assert m.count == 4

    def test_weighted_average_two_batches(self):
        """
        Batch 1: loss=1.0 over 8 samples.
        Batch 2: loss=3.0 over 2 samples.
        Correct weighted avg = (1.0*8 + 3.0*2) / 10 = 14/10 = 1.4.
        An unweighted mean of the two per-batch losses would give 2.0 — wrong.
        """
        m = AverageMeter()
        m.update(1.0, n=8)
        m.update(3.0, n=2)
        assert m.avg == pytest.approx(1.4)

    def test_reset_clears_state(self):
        m = AverageMeter()
        m.update(5.0, n=10)
        m.reset()
        assert m.avg == pytest.approx(0.0)
        assert m.count == 0
        assert m.total == pytest.approx(0.0)

    def test_n_defaults_to_one(self):
        """Default n=1 lets AverageMeter work as a plain running mean."""
        m = AverageMeter()
        for v in [1.0, 2.0, 3.0]:
            m.update(v)
        assert m.avg == pytest.approx(2.0)

    def test_accuracy_accumulation(self):
        """
        Accumulate batch-accuracy fractions to get epoch accuracy.
        Batch 1: 6/8 correct  → acc_fraction = 0.75, n=8
        Batch 2: 2/2 correct  → acc_fraction = 1.00, n=2
        Epoch acc = (6 + 2) / 10 = 0.80.
        """
        m = AverageMeter()
        m.update(6 / 8, n=8)
        m.update(2 / 2, n=2)
        assert m.avg == pytest.approx(0.80)

    def test_repr_contains_avg(self):
        m = AverageMeter()
        m.update(1.0)
        assert "avg=" in repr(m)


# ──────────────────────────────────────────────────────────────────────────────
# save_checkpoint / load_checkpoint
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_payload(base_cfg) -> dict:
    """Minimal valid checkpoint payload."""
    return {
        "epoch":           3,
        "model_state":     {"w": torch.tensor([1.0, 2.0])},
        "optimizer_state": {"state": {}, "param_groups": []},
        "scheduler_state": None,
        "metrics":         {"val_loss": 0.05, "val_acc": 0.99},
        "config":          base_cfg,
    }


class TestCheckpointIO:
    def test_roundtrip(self, sample_payload, tmp_path):
        """save + load preserves all fields."""
        path = tmp_path / "ckpt.pt"
        save_checkpoint(sample_payload, path)
        loaded = load_checkpoint(path)
        assert loaded["epoch"] == sample_payload["epoch"]
        assert torch.equal(loaded["model_state"]["w"], sample_payload["model_state"]["w"])
        assert loaded["metrics"] == sample_payload["metrics"]

    def test_parent_dirs_created(self, sample_payload, tmp_path):
        path = tmp_path / "deep" / "nested" / "ckpt.pt"
        save_checkpoint(sample_payload, path)
        assert path.exists()

    def test_file_exists_after_save(self, sample_payload, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(sample_payload, path)
        assert path.exists()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_load_missing_key_raises(self, tmp_path):
        """Checkpoint missing a required key raises KeyError with useful message."""
        incomplete = {"epoch": 1, "model_state": {}}   # missing required keys
        path = tmp_path / "bad.pt"
        torch.save(incomplete, path)
        with pytest.raises(KeyError, match="missing required keys"):
            load_checkpoint(path)

    def test_accepts_path_or_str(self, sample_payload, tmp_path):
        path = tmp_path / "ckpt.pt"
        save_checkpoint(sample_payload, str(path))     # str input
        loaded = load_checkpoint(str(path))            # str input
        assert loaded["epoch"] == 3

    def test_string_path_arg(self, sample_payload, tmp_path):
        """Both save and load accept string paths as well as Path objects."""
        path = str(tmp_path / "str_ckpt.pt")
        save_checkpoint(sample_payload, path)
        loaded = load_checkpoint(path, device="cpu")
        assert loaded["epoch"] == sample_payload["epoch"]


# ──────────────────────────────────────────────────────────────────────────────
# plot_training_curves
# ──────────────────────────────────────────────────────────────────────────────

def _make_metrics(n: int = 3) -> list[dict]:
    return [
        {"epoch": i + 1, "train_loss": 1.0 / (i + 1),
         "val_loss": 1.2 / (i + 1), "val_acc": 0.5 + i * 0.1}
        for i in range(n)
    ]


class TestPlotTrainingCurves:
    def test_png_created(self, tmp_path):
        """A PNG file is written at output_path."""
        out = tmp_path / "curves.png"
        plot_training_curves(_make_metrics(5), out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_parent_dirs_created(self, tmp_path):
        out = tmp_path / "sub" / "dir" / "curves.png"
        plot_training_curves(_make_metrics(3), out)
        assert out.exists()

    def test_empty_metrics_no_error(self, tmp_path):
        """Empty list produces no file but does not raise."""
        out = tmp_path / "empty.png"
        plot_training_curves([], out)
        assert not out.exists()

    def test_single_epoch(self, tmp_path):
        """Single-epoch metrics (no line visible, but no error)."""
        out = tmp_path / "single.png"
        plot_training_curves(_make_metrics(1), out)
        assert out.exists()

    def test_title_accepted(self, tmp_path):
        """Custom title does not cause an error."""
        out = tmp_path / "titled.png"
        plot_training_curves(_make_metrics(3), out, title="run_cnn_001")
        assert out.exists()


# ──────────────────────────────────────────────────────────────────────────────
# save_json / load_json
# ──────────────────────────────────────────────────────────────────────────────

class TestJsonHelpers:
    def test_roundtrip(self, tmp_path):
        data = {"a": 1, "b": [2, 3], "c": {"nested": True}}
        path = tmp_path / "out.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_sort_keys_produces_sorted_output(self, tmp_path):
        """sort_keys=True (default) makes keys appear in lexicographic order."""
        data = {"z": 3, "a": 1, "m": 2}
        path = tmp_path / "sorted.json"
        save_json(data, path)
        raw = path.read_text(encoding="utf-8")
        pos_a = raw.index('"a"')
        pos_m = raw.index('"m"')
        pos_z = raw.index('"z"')
        assert pos_a < pos_m < pos_z, "Keys are not in sorted order"

    def test_unsorted_keys_option(self, tmp_path):
        """sort_keys=False preserves insertion order (Python 3.7+ dict guarantee)."""
        data = {"z": 3, "a": 1}
        path = tmp_path / "unsorted.json"
        save_json(data, path, sort_keys=False)
        raw = path.read_text(encoding="utf-8")
        assert raw.index('"z"') < raw.index('"a"')

    def test_parent_dirs_created(self, tmp_path):
        path = tmp_path / "deep" / "dir" / "file.json"
        save_json({"x": 1}, path)
        assert path.exists()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "missing.json")

    def test_load_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_json(path)

    def test_indent_formatting(self, tmp_path):
        """indent=2 produces multi-line pretty-printed output."""
        path = tmp_path / "pretty.json"
        save_json({"key": "value"}, path, indent=2)
        content = path.read_text()
        assert "\n" in content

    def test_accepts_list(self, tmp_path):
        """Non-dict top-level values are also handled."""
        path = tmp_path / "list.json"
        save_json([1, 2, 3], path)
        assert load_json(path) == [1, 2, 3]


# ──────────────────────────────────────────────────────────────────────────────
# Timer
# ──────────────────────────────────────────────────────────────────────────────

class TestTimer:
    def test_context_manager_elapsed_positive(self):
        """Context manager records positive elapsed time."""
        with Timer() as t:
            _ = sum(range(10_000))
        assert t.elapsed > 0.0

    def test_context_manager_elapsed_set_after_exit(self):
        """elapsed is 0.0 before exit, positive after."""
        timer = Timer()
        with timer:
            pass
        assert timer.elapsed > 0.0

    def test_explicit_start_stop(self):
        t = Timer().start()
        _ = sum(range(10_000))
        elapsed = t.stop()
        assert elapsed > 0.0
        assert t.elapsed == pytest.approx(elapsed)

    def test_restart_resets_elapsed(self):
        """Calling start() again resets the timer."""
        t = Timer().start()
        time.sleep(0.01)
        _ = t.stop()
        first = t.elapsed

        t.start()
        # Don't do any work — elapsed should be near-zero
        t.stop()
        second = t.elapsed
        assert second < first

    def test_repr_contains_elapsed(self):
        with Timer() as t:
            pass
        assert "elapsed=" in repr(t)

    def test_chaining(self):
        """start() returns self for chaining."""
        t = Timer()
        result = t.start()
        assert result is t

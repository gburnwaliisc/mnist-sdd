"""
src/utils.py — cross-cutting utilities shared across the MNIST pipeline.

Sections:
    1. set_seed          — reproducible RNG initialisation
    2. configure_logging — root-logger setup with optional file sink
    3. AverageMeter      — sample-weighted running average
    4. save_checkpoint   — write SPEC §10.4.2 checkpoint dict to disk
       load_checkpoint   — read and validate a checkpoint dict
    5. plot_training_curves — standalone training/validation curve plotter
    6. save_json / load_json — deterministic JSON I/O helpers
    7. Timer             — context-manager and explicit start/stop timer

All functions accept pathlib.Path or str for file paths.
No module in src/ imports from this file at load time — callers import
only what they need, keeping individual module start-up costs low.

SPEC reference: SDD-MNIST-001 §8.6, §9, §10, RR-001..RR-007
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Seed management
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_CKPT_KEYS = frozenset(
    {"epoch", "model_state", "optimizer_state", "scheduler_state", "metrics", "config"}
)


def set_seed(seed: int) -> None:
    """
    Seed all RNG sources used in the pipeline for full reproducibility.

    Sets:
        Python random   — affects random.shuffle, random.choice, etc.
        NumPy           — affects sklearn's StratifiedShuffleSplit
        PyTorch CPU     — affects weight init, DataLoader worker order
        PyTorch CUDA    — affects GPU kernels when a GPU is available
        cuDNN flags     — deterministic=True, benchmark=False

    The cuDNN flags trade ~10 % GPU speed for guaranteed reproducibility:
    benchmark=True lets cuDNN pick a different kernel each run based on
    hardware state, which breaks the deterministic guarantee.

    Args:
        seed: Non-negative integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    logger.debug("RNG seeded with seed=%d", seed)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Logging configuration
# ──────────────────────────────────────────────────────────────────────────────

def configure_logging(
    level: str | int = "INFO",
    log_file: Path | str | None = None,
    fmt: str = "%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt: str = "%H:%M:%S",
) -> logging.Logger:
    """
    Configure the root logger with a console handler and an optional file handler.

    Idempotent: if the root logger already has handlers, the function adds only
    the handlers that are not already present (identified by handler type and
    file path).  This prevents handler duplication when multiple Optuna trials
    run in the same process and each calls configure_logging.

    Args:
        level:    Log level name ('DEBUG', 'INFO', 'WARNING', 'ERROR') or
                  the corresponding integer constant.
        log_file: Optional path for a FileHandler sink.  Created (with parent
                  directories) if it does not exist.
        fmt:      Log record format string.
        datefmt:  Timestamp format for %(asctime)s.

    Returns:
        The configured root logger.
    """
    root = logging.getLogger()
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    root.setLevel(level)

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler — add only if no StreamHandler is already present
    has_stream = any(isinstance(h, logging.StreamHandler) and
                     not isinstance(h, logging.FileHandler)
                     for h in root.handlers)
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        root.addHandler(sh)

    # File handler — add only if a different file is not already logged to
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        existing_paths = {
            Path(h.baseFilename)
            for h in root.handlers
            if isinstance(h, logging.FileHandler)
        }
        if log_file.resolve() not in {p.resolve() for p in existing_paths}:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            root.addHandler(fh)

    return root


# ──────────────────────────────────────────────────────────────────────────────
# 3. Metric tracking
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """
    Track a sample-weighted running average over batches.

    Usage::

        meter = AverageMeter()
        for images, labels in loader:
            loss = criterion(model(images), labels)
            meter.update(loss.item(), n=labels.size(0))
        epoch_loss = meter.avg

    The ``n`` argument (batch size) is critical for correctness when the
    final batch is smaller than a full batch.  A naive unweighted mean of
    per-batch values would under-weight the last batch.

    Attributes:
        total: Weighted sum of all values seen so far.
        count: Total number of samples accumulated.
        avg:   Weighted mean (total / count), or 0.0 if count == 0.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self.total: float = 0.0
        self.count: int   = 0

    def update(self, value: float, n: int = 1) -> None:
        """
        Accumulate a new observation.

        Args:
            value: The metric value for this batch (e.g. mean loss over batch).
            n:     Number of samples in this batch.  Default 1 for non-loss
                   metrics (e.g., binary correct/wrong per sample).
        """
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """Weighted average, or 0.0 if no samples have been accumulated."""
        return self.total / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(avg={self.avg:.6f}, count={self.count})"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Checkpoint I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(payload: dict, path: Path | str) -> None:
    """
    Write a checkpoint dict to disk atomically.

    The checkpoint is first written to a temporary file in the same directory,
    then renamed over the target path.  This prevents a half-written file from
    being read if the process is interrupted mid-write — rename is atomic on
    POSIX and approximately atomic on Windows (same drive, same filesystem).

    The caller is responsible for constructing ``payload`` to match the
    schema defined in SPEC §10.4.2::

        {
            'epoch':           int,
            'model_state':     dict,
            'optimizer_state': dict,
            'scheduler_state': dict | None,
            'metrics':         dict,
            'config':          dict,
        }

    Args:
        payload: Dict conforming to the checkpoint schema.
        path:    Destination path (created with parent dirs if needed).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    logger.debug("Checkpoint saved to %s", path)


def load_checkpoint(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> dict:
    """
    Load and validate a checkpoint written by :func:`save_checkpoint`.

    Args:
        path:   Path to the .pt file.
        device: Device to map tensors to.  Use 'cpu' when loading on a
                machine that may not have a GPU.

    Returns:
        Checkpoint dict.  All required keys are present.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError:          If a required checkpoint key is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=False: checkpoint contains non-tensor objects (config dict,
    # metrics dict).  Source is always this codebase — trusted.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    missing = _REQUIRED_CKPT_KEYS - ckpt.keys()
    if missing:
        raise KeyError(
            f"Checkpoint at '{path}' is missing required keys: {sorted(missing)}. "
            "The file may have been written by an incompatible version."
        )

    logger.debug("Checkpoint loaded from %s (epoch %d)", path, ckpt["epoch"])
    return ckpt


# ──────────────────────────────────────────────────────────────────────────────
# 5. Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    epoch_metrics: list[dict],
    output_path: Path | str,
    title: str = "",
) -> None:
    """
    Plot loss curves and validation accuracy from per-epoch metric dicts.

    Each dict in ``epoch_metrics`` must contain the keys produced by
    Trainer: ``epoch``, ``train_loss``, ``val_loss``, ``val_acc``.

    Writes a PNG to ``output_path``.  Uses the Agg backend (no display
    server required), which is set at module import time.

    Args:
        epoch_metrics: List of per-epoch metric dicts (one per epoch).
        output_path:   Destination PNG path.
        title:         Optional super-title (e.g. the run_id).
    """
    if not epoch_metrics:
        logger.warning("plot_training_curves: no metrics to plot.")
        return

    epochs     = [m["epoch"]      for m in epoch_metrics]
    train_loss = [m["train_loss"] for m in epoch_metrics]
    val_loss   = [m["val_loss"]   for m in epoch_metrics]
    val_acc    = [m["val_acc"]    for m in epoch_metrics]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, train_loss, label="Train loss",      linewidth=1.5)
    ax_loss.plot(epochs, val_loss,   label="Val loss",        linewidth=1.5)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss curves")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.4)

    ax_acc.plot(epochs, val_acc, color="tab:green", label="Val accuracy", linewidth=1.5)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Validation accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.4)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Training curves saved to %s", output_path)


# ──────────────────────────────────────────────────────────────────────────────
# 6. JSON helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_json(
    data: Any,
    path: Path | str,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    """
    Serialise ``data`` to a JSON file.

    ``sort_keys=True`` (the default) produces deterministic key ordering,
    making the output git-diffable: if two runs produce identical metrics,
    the JSON files are byte-for-byte identical and ``git diff`` shows no change.

    Parent directories are created automatically.

    Args:
        data:      JSON-serialisable object (dict, list, scalar).
        path:      Destination file path.
        indent:    Pretty-print indentation level (spaces).
        sort_keys: Write object keys in sorted order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)
    logger.debug("JSON saved to %s", path)


def load_json(path: Path | str) -> Any:
    """
    Load a JSON file and return the parsed object.

    Args:
        path: Path to the JSON file.

    Returns:
        The deserialised Python object (usually a dict or list).

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Timer
# ──────────────────────────────────────────────────────────────────────────────

class Timer:
    """
    Measure elapsed wall-clock time.  Usable as a context manager or
    via explicit ``start()`` / ``stop()`` calls.

    Context manager usage::

        with Timer() as t:
            run_training_epoch(...)
        print(f"Epoch took {t.elapsed:.2f}s")

    Explicit usage::

        t = Timer().start()
        do_work()
        elapsed = t.stop()   # also stored in t.elapsed

    Attributes:
        elapsed: Seconds elapsed between start and stop (float).
                 0.0 before the first stop() or __exit__ call.
    """

    def __init__(self) -> None:
        self._t0:    float = 0.0
        self.elapsed: float = 0.0

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self._t0

    # ── Explicit interface ───────────────────────────────────────────────────

    def start(self) -> "Timer":
        """Start (or restart) the timer.  Returns self for chaining."""
        self._t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def stop(self) -> float:
        """Stop the timer.  Returns elapsed seconds and stores in self.elapsed."""
        self.elapsed = time.perf_counter() - self._t0
        return self.elapsed

    def __repr__(self) -> str:
        return f"Timer(elapsed={self.elapsed:.4f}s)"

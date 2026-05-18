"""
Configuration loader for the MNIST SDD project.

Public API:
    load_config(path, overrides=None) -> dict

SPEC reference: SDD-MNIST-001 §14 (FR-CFG-001–003, AC-CFG-001–003)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ── Required key sets ─────────────────────────────────────────────────────────

_REQUIRED_TOP      = {"seed", "data", "model", "training"}
_REQUIRED_DATA     = {"root", "batch_size", "val_split", "normalize"}
_REQUIRED_NORM     = {"mean", "std"}
_REQUIRED_MODEL    = {"arch", "num_classes", "dropout", "activation"}
_REQUIRED_TRAINING = {"epochs", "optimizer", "learning_rate", "weight_decay",
                      "checkpoint"}
_REQUIRED_CKPT     = {"dir"}

_VALID_ARCHES      = {"cnn", "mlp"}
_VALID_OPTIMIZERS  = {"adam", "sgd"}
_VALID_SCHEDULERS  = {"none", "step", "cosine", "plateau"}
_VALID_ACTIVATIONS = {"relu", "gelu", "tanh"}
_VALID_ES_MONITORS = {"val_loss", "val_acc"}


# ── Public API ────────────────────────────────────────────────────────────────

def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict:
    """
    Load a YAML config file, apply optional dot-path overrides, and validate.

    Args:
        path:      Path to the YAML configuration file.
        overrides: Optional flat dict of dot-notation key paths to new values.
                   Example: {"training.epochs": 50, "model.dropout": 0.3}
                   Nested keys are separated by '.'.

    Returns:
        Validated config dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError:        If required keys are absent or values are invalid.
        yaml.YAMLError:    If the file contains invalid YAML syntax.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()}"
        )

    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")

    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    _validate(cfg)
    return cfg


# ── Override helpers ──────────────────────────────────────────────────────────

def _apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """
    Apply flat dot-path overrides to a nested dict.

    The input is deep-copied; the original is never mutated.

        cfg = {'training': {'epochs': 20}}
        overrides = {'training.epochs': 50}
        → result: {'training': {'epochs': 50}}
    """
    cfg = copy.deepcopy(cfg)
    for key_path, value in overrides.items():
        keys   = key_path.split(".")
        target = cfg
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
    return cfg


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(cfg: dict) -> None:
    """
    Check required keys and value constraints.

    Raises ValueError with a descriptive message on the first violation.
    Optional/tuning sub-keys are not validated — they fall back to defaults
    at the call site via dict.get().
    """
    _require_keys(cfg, _REQUIRED_TOP, "config")

    if not isinstance(cfg["seed"], int):
        raise ValueError("config.seed must be an integer.")

    # data ────────────────────────────────────────────────────────────────────
    d = cfg["data"]
    _require_keys(d, _REQUIRED_DATA, "data")
    _require_keys(d["normalize"], _REQUIRED_NORM, "data.normalize")

    val_split = d["val_split"]
    if not (0.0 < val_split < 1.0):
        raise ValueError(
            f"data.val_split must be strictly in (0, 1); got {val_split!r}."
        )

    # model ───────────────────────────────────────────────────────────────────
    m = cfg["model"]
    _require_keys(m, _REQUIRED_MODEL, "model")

    arch = m["arch"]
    if arch not in _VALID_ARCHES:
        raise ValueError(
            f"model.arch must be one of {sorted(_VALID_ARCHES)}; "
            f"got {arch!r}."
        )

    act = m["activation"]
    if act not in _VALID_ACTIVATIONS:
        raise ValueError(
            f"model.activation must be one of {sorted(_VALID_ACTIVATIONS)}; "
            f"got {act!r}."
        )

    dropout = m["dropout"]
    if not (0.0 <= dropout < 1.0):
        raise ValueError(
            f"model.dropout must be in [0, 1); got {dropout!r}."
        )

    # training ────────────────────────────────────────────────────────────────
    t = cfg["training"]
    _require_keys(t, _REQUIRED_TRAINING, "training")
    _require_keys(t["checkpoint"], _REQUIRED_CKPT, "training.checkpoint")

    opt = t["optimizer"].lower()
    if opt not in _VALID_OPTIMIZERS:
        raise ValueError(
            f"training.optimizer must be one of {sorted(_VALID_OPTIMIZERS)}; "
            f"got {t['optimizer']!r}."
        )

    if "lr_scheduler" in t:
        sched = t["lr_scheduler"].get("type", "none").lower()
        if sched not in _VALID_SCHEDULERS:
            raise ValueError(
                f"training.lr_scheduler.type must be one of "
                f"{sorted(_VALID_SCHEDULERS)}; got {sched!r}."
            )

    if "early_stopping" in t:
        monitor = t["early_stopping"].get("monitor", "val_loss")
        if monitor not in _VALID_ES_MONITORS:
            raise ValueError(
                f"training.early_stopping.monitor must be one of "
                f"{sorted(_VALID_ES_MONITORS)}; got {monitor!r}."
            )

    label_smoothing = t.get("label_smoothing", 0.0)
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(
            f"training.label_smoothing must be in [0, 1); "
            f"got {label_smoothing!r}."
        )


def _require_keys(d: dict, required: set[str], section: str) -> None:
    missing = required - d.keys()
    if missing:
        raise ValueError(
            f"config['{section}'] is missing required keys: {sorted(missing)}"
        )

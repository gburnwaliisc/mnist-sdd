"""
Tests for src/config.py.

All tests are self-contained; they write temporary YAML files to tmp_path.
No network access, no MNIST download.

Covers:
    AC-CFG-001  load_config returns a dict without raising (valid YAML)
    AC-CFG-002  load_config raises FileNotFoundError for a missing path
    AC-CFG-003  round-trip: saved config.yaml is re-loadable as an equal dict

Additional coverage:
    - Missing required top-level key raises ValueError
    - Invalid model.arch raises ValueError
    - Invalid data.val_split raises ValueError
    - Invalid model.dropout raises ValueError
    - Invalid training.optimizer raises ValueError
    - Invalid lr_scheduler.type raises ValueError
    - Invalid early_stopping.monitor raises ValueError
    - Invalid label_smoothing raises ValueError
    - Empty YAML file raises ValueError
    - Dot-path overrides are applied correctly
    - Overrides do not mutate the original config on disk
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.config import load_config, _apply_overrides


# ---------------------------------------------------------------------------
# Minimal valid config fixture
# ---------------------------------------------------------------------------

_MINIMAL_CFG = {
    "seed": 42,
    "data": {
        "root": "./data",
        "batch_size": 128,
        "val_split": 0.1,
        "augmentation": True,
        "normalize": {"mean": 0.1307, "std": 0.3081},
    },
    "model": {
        "arch": "cnn",
        "num_classes": 10,
        "dropout": 0.25,
        "activation": "relu",
        "conv_channels": [32, 64],
        "fc_hidden": 512,
        "use_batchnorm": True,
    },
    "training": {
        "epochs": 10,
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "checkpoint": {"dir": "./checkpoints"},
    },
}


def _write_cfg(path: Path, cfg: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh)


@pytest.fixture()
def valid_cfg_path(tmp_path) -> Path:
    p = tmp_path / "cfg.yaml"
    _write_cfg(p, _MINIMAL_CFG)
    return p


# ---------------------------------------------------------------------------
# AC-CFG-001 — valid YAML loads without error
# ---------------------------------------------------------------------------

class TestLoadConfigValid:
    def test_returns_dict(self, valid_cfg_path):
        """AC-CFG-001: load_config returns a dict."""
        cfg = load_config(valid_cfg_path)
        assert isinstance(cfg, dict)

    def test_top_level_keys_present(self, valid_cfg_path):
        """AC-CFG-001: required top-level keys are present in the result."""
        cfg = load_config(valid_cfg_path)
        for key in ("seed", "data", "model", "training"):
            assert key in cfg, f"Missing top-level key: {key!r}"

    def test_seed_value_preserved(self, valid_cfg_path):
        cfg = load_config(valid_cfg_path)
        assert cfg["seed"] == 42

    def test_nested_values_accessible(self, valid_cfg_path):
        cfg = load_config(valid_cfg_path)
        assert cfg["data"]["val_split"] == pytest.approx(0.1)
        assert cfg["model"]["arch"] == "cnn"

    def test_default_yaml_is_loadable(self):
        """AC-CFG-001: the shipped config/default.yaml loads without error."""
        default_path = Path("config/default.yaml")
        if not default_path.exists():
            pytest.skip("config/default.yaml not found (run from project root)")
        cfg = load_config(default_path)
        assert isinstance(cfg, dict)
        assert cfg["seed"] == 42


# ---------------------------------------------------------------------------
# AC-CFG-002 — missing file raises FileNotFoundError
# ---------------------------------------------------------------------------

class TestLoadConfigMissing:
    def test_missing_file_raises(self, tmp_path):
        """AC-CFG-002: FileNotFoundError for a path that does not exist."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_error_message_contains_path(self, tmp_path):
        """FileNotFoundError message includes the requested path."""
        p = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError, match="missing.yaml"):
            load_config(p)


# ---------------------------------------------------------------------------
# AC-CFG-003 — round-trip: saved config.yaml re-loads as equal dict
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    def test_yaml_dump_and_reload(self, tmp_path):
        """AC-CFG-003: dump → reload produces an identical dict."""
        p = tmp_path / "roundtrip.yaml"
        _write_cfg(p, _MINIMAL_CFG)
        cfg = load_config(p)
        # Re-dump and re-load
        p2 = tmp_path / "roundtrip2.yaml"
        with p2.open("w") as fh:
            yaml.dump(cfg, fh)
        cfg2 = load_config(p2)
        assert cfg == cfg2


# ---------------------------------------------------------------------------
# Validation — required keys
# ---------------------------------------------------------------------------

class TestValidationRequiredKeys:
    @pytest.mark.parametrize("missing_key", ["seed", "data", "model", "training"])
    def test_missing_top_level_key_raises(self, tmp_path, missing_key):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        del cfg[missing_key]
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match=missing_key):
            load_config(p)

    def test_missing_data_root_raises(self, tmp_path):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        del cfg["data"]["root"]
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match="root"):
            load_config(p)

    def test_missing_normalize_mean_raises(self, tmp_path):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        del cfg["data"]["normalize"]["mean"]
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match="mean"):
            load_config(p)

    def test_missing_training_checkpoint_raises(self, tmp_path):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        del cfg["training"]["checkpoint"]
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match="checkpoint"):
            load_config(p)

    def test_empty_yaml_raises(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(p)


# ---------------------------------------------------------------------------
# Validation — value constraints
# ---------------------------------------------------------------------------

class TestValidationValues:
    def _write_and_load(self, tmp_path, cfg):
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        return load_config(p)

    def test_invalid_arch_raises(self, tmp_path):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        cfg["model"]["arch"] = "transformer"
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match="arch"):
            load_config(p)

    def test_valid_arch_mlp(self, tmp_path):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        cfg["model"]["arch"] = "mlp"
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        loaded = load_config(p)
        assert loaded["model"]["arch"] == "mlp"

    def test_invalid_activation_raises(self, tmp_path):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        cfg["model"]["activation"] = "sigmoid"
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match="activation"):
            load_config(p)

    @pytest.mark.parametrize("bad_val", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_val_split_raises(self, tmp_path, bad_val):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        cfg["data"]["val_split"] = bad_val
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match="val_split"):
            load_config(p)

    @pytest.mark.parametrize("bad_val", [-0.1, 1.0, 1.5])
    def test_invalid_dropout_raises(self, tmp_path, bad_val):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        cfg["model"]["dropout"] = bad_val
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match="dropout"):
            load_config(p)

    def test_dropout_zero_allowed(self, tmp_path):
        """dropout=0.0 (no dropout) is a valid configuration."""
        cfg = copy.deepcopy(_MINIMAL_CFG)
        cfg["model"]["dropout"] = 0.0
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        loaded = load_config(p)
        assert loaded["model"]["dropout"] == 0.0

    def test_invalid_optimizer_raises(self, tmp_path):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        cfg["training"]["optimizer"] = "rmsprop"
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match="optimizer"):
            load_config(p)

    def test_invalid_scheduler_type_raises(self, tmp_path):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        cfg["training"]["lr_scheduler"] = {"type": "warmup"}
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match="lr_scheduler"):
            load_config(p)

    def test_invalid_es_monitor_raises(self, tmp_path):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        cfg["training"]["early_stopping"] = {"monitor": "test_loss"}
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match="monitor"):
            load_config(p)

    def test_invalid_label_smoothing_raises(self, tmp_path):
        cfg = copy.deepcopy(_MINIMAL_CFG)
        cfg["training"]["label_smoothing"] = 1.5
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        with pytest.raises(ValueError, match="label_smoothing"):
            load_config(p)

    def test_label_smoothing_zero_allowed(self, tmp_path):
        """label_smoothing=0.0 (standard cross-entropy) is valid."""
        cfg = copy.deepcopy(_MINIMAL_CFG)
        cfg["training"]["label_smoothing"] = 0.0
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, cfg)
        loaded = load_config(p)
        assert loaded["training"]["label_smoothing"] == 0.0


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

class TestApplyOverrides:
    def test_single_nested_override(self, valid_cfg_path):
        cfg = load_config(valid_cfg_path, overrides={"training.epochs": 99})
        assert cfg["training"]["epochs"] == 99

    def test_deep_nested_override(self, valid_cfg_path):
        cfg = load_config(valid_cfg_path, overrides={"data.normalize.mean": 0.5})
        assert cfg["data"]["normalize"]["mean"] == pytest.approx(0.5)

    def test_multiple_overrides(self, valid_cfg_path):
        cfg = load_config(
            valid_cfg_path,
            overrides={"model.dropout": 0.1, "training.learning_rate": 1e-4},
        )
        assert cfg["model"]["dropout"] == pytest.approx(0.1)
        assert cfg["training"]["learning_rate"] == pytest.approx(1e-4)

    def test_override_does_not_mutate_original(self, tmp_path):
        """Overrides deep-copy the config; the YAML file is not modified."""
        p = tmp_path / "cfg.yaml"
        _write_cfg(p, _MINIMAL_CFG)
        load_config(p, overrides={"training.epochs": 999})
        # Reload from disk — should still have original value
        reloaded = load_config(p)
        assert reloaded["training"]["epochs"] == _MINIMAL_CFG["training"]["epochs"]

    def test_apply_overrides_direct(self):
        """_apply_overrides helper does not mutate its input."""
        original = {"a": {"b": 1}}
        result = _apply_overrides(original, {"a.b": 99})
        assert original["a"]["b"] == 1  # original unchanged
        assert result["a"]["b"] == 99

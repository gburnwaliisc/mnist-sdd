"""
Tests for src/model/cnn.py — MNISTConvNet.

All tests use synthetic tensors; no dataset download or file I/O.

Covers acceptance criteria:
    AC-MDL-001  forward output shape is (B, 10)
    AC-MDL-003  eval() mode: identical outputs on repeated passes (dropout off)
    AC-MDL-004  number of conv blocks matches len(conv_channels)

Additional coverage:
    - Custom conv_channels depth and widths
    - GELU activation variant
    - BatchNorm disabled (use_batchnorm=False)
    - No NaN / Inf in initial weights or first forward pass
    - count_parameters returns positive integers
    - Invalid config raises ValueError with informative message
    - state_dict keys follow expected naming convention
    - Logits are raw (no softmax applied inside model)
    - Gradient flow: loss.backward() completes without error
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.model.cnn import MNISTConvNet


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def default_model(base_cfg) -> MNISTConvNet:
    """MNISTConvNet with the default config (conv_channels=[32,64], relu)."""
    return MNISTConvNet(base_cfg)


@pytest.fixture()
def batch() -> torch.Tensor:
    """Synthetic batch of 4 greyscale 28×28 images."""
    return torch.randn(4, 1, 28, 28)


# ──────────────────────────────────────────────────────────────────────────────
# Output shape (AC-MDL-001)
# ──────────────────────────────────────────────────────────────────────────────

class TestOutputShape:
    def test_default_output_shape(self, default_model, batch):
        """AC-MDL-001: forward returns (B, 10) for default config."""
        logits = default_model(batch)
        assert logits.shape == (4, 10), (
            f"Expected (4, 10), got {logits.shape}"
        )

    def test_batch_size_one(self, default_model):
        """Single-sample inference (B=1) returns (1, 10)."""
        x = torch.randn(1, 1, 28, 28)
        assert default_model(x).shape == (1, 10)

    def test_large_batch(self, default_model):
        """Large batch (B=128) returns (128, 10)."""
        x = torch.randn(128, 1, 28, 28)
        assert default_model(x).shape == (128, 10)

    def test_output_is_raw_logits(self, default_model, batch):
        """
        Model output must be raw logits, not probabilities.

        Probabilities would sum to 1 across the class dimension.
        Raw logits sum to an arbitrary value.
        """
        logits = default_model(batch)
        row_sums = logits.sum(dim=1)
        # If this were softmax output, all rows would sum to ≈ 1.0
        # For logits, at least some rows should differ substantially from 1.0
        not_all_one = not torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.1)
        assert not_all_one, (
            "All rows sum to ~1.0 — model may be applying softmax internally"
        )


# ──────────────────────────────────────────────────────────────────────────────
# eval() / train() discipline (AC-MDL-003)
# ──────────────────────────────────────────────────────────────────────────────

class TestEvalMode:
    def test_eval_mode_deterministic(self, default_model, batch):
        """
        AC-MDL-003: repeated forward passes in eval() mode yield
        identical outputs (Dropout deactivated, BN uses running stats).
        """
        default_model.eval()
        with torch.no_grad():
            out1 = default_model(batch)
            out2 = default_model(batch)
        assert torch.equal(out1, out2), (
            "Outputs differ across eval() passes — Dropout may still be active"
        )

    def test_train_mode_stochastic_with_dropout(self, base_cfg, batch):
        """
        In train() mode with high dropout (p=0.9), outputs should
        differ across forward passes (non-deterministic due to masking).
        """
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["dropout"] = 0.9
        model = MNISTConvNet(cfg)
        model.train()

        out1 = model(batch)
        out2 = model(batch)
        # With p=0.9 dropout, the probability of all outputs being identical
        # is astronomically small for a 256-dimensional hidden layer.
        assert not torch.equal(out1, out2), (
            "Training-mode outputs are identical — Dropout may be inactive"
        )

    def test_model_training_flag(self, default_model):
        """model.training reflects the current mode."""
        default_model.train()
        assert default_model.training is True

        default_model.eval()
        assert default_model.training is False


# ──────────────────────────────────────────────────────────────────────────────
# Conv block count (AC-MDL-004)
# ──────────────────────────────────────────────────────────────────────────────

class TestConvBlockCount:
    @pytest.mark.parametrize("channels", [
        [16],
        [32, 64],
        [16, 32, 64],
    ])
    def test_conv_block_count_matches_channels(self, base_cfg, channels):
        """
        AC-MDL-004: number of Conv2d layers in conv_blocks equals
        len(conv_channels).
        """
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["conv_channels"] = channels
        model = MNISTConvNet(cfg)

        conv_layers = [
            m for m in model.conv_blocks.modules()
            if isinstance(m, nn.Conv2d)
        ]
        assert len(conv_layers) == len(channels), (
            f"Expected {len(channels)} Conv2d layers for channels={channels}; "
            f"found {len(conv_layers)}"
        )

    @pytest.mark.parametrize("channels", [
        [16],
        [32, 64],
        [16, 32, 64],
    ])
    def test_output_shape_with_variable_depth(self, base_cfg, batch, channels):
        """Variable-depth conv stacks all produce (B, 10) logits."""
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["conv_channels"] = channels
        model = MNISTConvNet(cfg)
        assert model(batch).shape == (4, 10)


# ──────────────────────────────────────────────────────────────────────────────
# Activation variants
# ──────────────────────────────────────────────────────────────────────────────

class TestActivations:
    def test_gelu_forward_shape(self, base_cfg, batch):
        """GELU activation produces correct output shape."""
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["activation"] = "gelu"
        model = MNISTConvNet(cfg)
        assert model(batch).shape == (4, 10)

    def test_invalid_activation_raises(self, base_cfg):
        """ValueError is raised for an unrecognised activation name."""
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["activation"] = "sigmoid"
        with pytest.raises(ValueError, match="Unsupported activation"):
            MNISTConvNet(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# BatchNorm toggle
# ──────────────────────────────────────────────────────────────────────────────

class TestBatchNorm:
    def test_batchnorm_disabled_output_shape(self, base_cfg, batch):
        """use_batchnorm=False still produces (B, 10) output."""
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["use_batchnorm"] = False
        model = MNISTConvNet(cfg)
        assert model(batch).shape == (4, 10)

    def test_batchnorm_present_when_enabled(self, default_model):
        """BatchNorm2d layers exist in conv_blocks when use_batchnorm=True."""
        bn_layers = [
            m for m in default_model.conv_blocks.modules()
            if isinstance(m, nn.BatchNorm2d)
        ]
        assert len(bn_layers) > 0, "Expected BatchNorm2d layers when use_batchnorm=True"

    def test_batchnorm_absent_when_disabled(self, base_cfg):
        """No BatchNorm2d layers when use_batchnorm=False."""
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["use_batchnorm"] = False
        model = MNISTConvNet(cfg)
        bn_layers = [
            m for m in model.conv_blocks.modules()
            if isinstance(m, nn.BatchNorm2d)
        ]
        assert len(bn_layers) == 0

    def test_conv_bias_absent_with_batchnorm(self, default_model):
        """
        Conv2d layers have no bias when BatchNorm follows — the BN β
        parameter subsumes the conv bias.
        """
        for m in default_model.conv_blocks.modules():
            if isinstance(m, nn.Conv2d):
                assert m.bias is None, (
                    "Conv2d should have bias=False when BatchNorm follows"
                )

    def test_conv_bias_present_without_batchnorm(self, base_cfg):
        """Conv2d has bias when use_batchnorm=False."""
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["use_batchnorm"] = False
        model = MNISTConvNet(cfg)
        for m in model.conv_blocks.modules():
            if isinstance(m, nn.Conv2d):
                assert m.bias is not None


# ──────────────────────────────────────────────────────────────────────────────
# Weight initialisation
# ──────────────────────────────────────────────────────────────────────────────

class TestWeightInit:
    def test_no_nan_in_weights(self, default_model):
        """Initial weights contain no NaN values."""
        for name, param in default_model.named_parameters():
            assert not torch.isnan(param).any(), (
                f"NaN found in initial weights of '{name}'"
            )

    def test_no_inf_in_weights(self, default_model):
        """Initial weights contain no Inf values."""
        for name, param in default_model.named_parameters():
            assert not torch.isinf(param).any(), (
                f"Inf found in initial weights of '{name}'"
            )

    def test_batchnorm_initial_weight_one(self, default_model):
        """BatchNorm2d weight (γ) initialised to 1."""
        for m in default_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert torch.allclose(m.weight, torch.ones_like(m.weight)), (
                    "BatchNorm2d weight not initialised to 1"
                )

    def test_batchnorm_initial_bias_zero(self, default_model):
        """BatchNorm2d bias (β) initialised to 0."""
        for m in default_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert torch.allclose(m.bias, torch.zeros_like(m.bias)), (
                    "BatchNorm2d bias not initialised to 0"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Gradient flow
# ──────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:
    def test_backward_completes(self, default_model, batch):
        """loss.backward() runs without error; all gradients are finite."""
        default_model.train()
        labels = torch.randint(0, 10, (4,))
        logits = default_model(batch)
        loss   = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        for name, param in default_model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient in '{name}'"
                )
                assert not torch.isinf(param.grad).any(), (
                    f"Inf gradient in '{name}'"
                )

    def test_no_grad_in_eval(self, default_model, batch):
        """torch.no_grad() context produces tensors with requires_grad=False."""
        default_model.eval()
        with torch.no_grad():
            logits = default_model(batch)
        assert not logits.requires_grad


# ──────────────────────────────────────────────────────────────────────────────
# Parameter counts and state_dict
# ──────────────────────────────────────────────────────────────────────────────

class TestIntrospection:
    def test_count_parameters_positive(self, default_model):
        """count_parameters returns positive integers."""
        stats = default_model.count_parameters()
        assert stats["total"] > 0
        assert stats["trainable"] > 0
        assert stats["trainable"] <= stats["total"]

    def test_state_dict_key_naming(self, default_model):
        """
        state_dict keys follow the named-block convention:
            conv_blocks.conv1.weight
            conv_blocks.bn1.weight
            fc_head.fc1.weight
            fc_head.fc2.weight
        """
        keys = set(default_model.state_dict().keys())
        assert "conv_blocks.conv1.weight" in keys, (
            "Expected key 'conv_blocks.conv1.weight'"
        )
        assert "conv_blocks.conv2.weight" in keys, (
            "Expected key 'conv_blocks.conv2.weight'"
        )
        assert "fc_head.fc1.weight" in keys, (
            "Expected key 'fc_head.fc1.weight'"
        )
        assert "fc_head.fc2.weight" in keys, (
            "Expected key 'fc_head.fc2.weight'"
        )

    def test_fc_input_size_attribute(self, default_model):
        """fc_input_size attribute equals 64 × 7 × 7 = 3136 for default config."""
        assert default_model.fc_input_size == 3136, (
            f"Expected fc_input_size=3136, got {default_model.fc_input_size}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Invalid config
# ──────────────────────────────────────────────────────────────────────────────

class TestInvalidConfig:
    def test_empty_conv_channels_raises(self, base_cfg):
        """ValueError when conv_channels is empty."""
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["conv_channels"] = []
        with pytest.raises(ValueError, match="conv_channels"):
            MNISTConvNet(cfg)

    def test_dropout_out_of_range_raises(self, base_cfg):
        """ValueError when dropout >= 1.0."""
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["dropout"] = 1.0
        with pytest.raises(ValueError, match="dropout"):
            MNISTConvNet(cfg)

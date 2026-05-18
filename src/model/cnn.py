"""
MNISTConvNet — configurable convolutional neural network for MNIST.

Architecture overview (default config: conv_channels=[32,64], fc_hidden=256)
─────────────────────────────────────────────────────────────────────────────
Input              (B,  1, 28, 28)   single-channel greyscale image

Conv Block 1       Conv2d(1→32, k=3, pad=1)   → (B, 32, 28, 28)
                   BatchNorm2d(32)             → (B, 32, 28, 28)
                   ReLU                        → (B, 32, 28, 28)
                   MaxPool2d(2, 2)             → (B, 32, 14, 14)

Conv Block 2       Conv2d(32→64, k=3, pad=1)  → (B, 64, 14, 14)
                   BatchNorm2d(64)             → (B, 64, 14, 14)
                   ReLU                        → (B, 64, 14, 14)
                   MaxPool2d(2, 2)             → (B, 64,  7,  7)

Flatten                                        → (B, 3136)

FC head            Linear(3136 → 256)          → (B, 256)
                   ReLU                        → (B, 256)
                   Dropout(p=0.5)             → (B, 256)
                   Linear(256 → 10)            → (B,  10)  raw logits

The number of conv blocks equals len(conv_channels); depth is fully
configurable without modifying this file.

SPEC reference: SDD-MNIST-001 §7.2 (FR-MDL-001, FR-MDL-004–006)
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Activation registry
# ──────────────────────────────────────────────────────────────────────────────

# Maps config string → nn.Module class.
# Adding a new activation: insert one line here, no other changes needed.
_ACTIVATION_CLASSES: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


def _make_activation(name: str) -> nn.Module:
    """
    Instantiate an activation module by config name.

    A new instance is returned on every call.  Sharing a single instance
    across layers is unsafe when in-place operations are involved — the
    same Module object appearing twice in a Sequential can cause autograd
    to trace through the same node twice, producing incorrect gradients.

    Args:
        name: Activation identifier from config.  Must be in
              {'relu', 'gelu'}.

    Returns:
        Fresh nn.Module instance.

    Raises:
        ValueError: If name is not a recognised activation.
    """
    if name not in _ACTIVATION_CLASSES:
        valid = sorted(_ACTIVATION_CLASSES)
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Valid choices: {valid}"
        )
    return _ACTIVATION_CLASSES[name]()


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class MNISTConvNet(nn.Module):
    """
    Configurable CNN for 10-class MNIST digit classification.

    The model is constructed entirely from cfg['model'], making it
    self-describing: any checkpoint that embeds the config used during
    training can be reconstructed without consulting external files.

    Output contract
    ~~~~~~~~~~~~~~~
    forward() returns raw logits of shape (B, 10).  No softmax is applied
    inside the model.  The caller is responsible for applying:
        - nn.CrossEntropyLoss(logits, targets)  during training
        - F.softmax(logits, dim=1)              during inference

    eval() / train() discipline
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Call model.eval() before any validation, test, or inference pass.
    This switches BatchNorm2d from batch statistics to running statistics
    and sets Dropout probability to zero.  Call model.train() at the
    start of each training epoch to restore training behaviour.

    Args:
        cfg: Full config dict.  Reads the cfg['model'] sub-dict.
             See SPEC §7.2.2 for the full parameter table.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg["model"]

        # ── Read hyperparameters ─────────────────────────────────────────────
        conv_channels: list[int] = m["conv_channels"]  # e.g. [32, 64]
        fc_hidden:     int       = m["fc_hidden"]       # e.g. 256
        dropout:       float     = m["dropout"]         # e.g. 0.5
        use_batchnorm: bool      = m["use_batchnorm"]   # True
        activation:    str       = m["activation"]      # "relu" | "gelu"
        num_classes:   int       = m["num_classes"]     # 10

        # ── Validate ─────────────────────────────────────────────────────────
        if len(conv_channels) < 1:
            raise ValueError(
                "model.conv_channels must contain at least one value; "
                f"got {conv_channels!r}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"model.dropout must be in [0, 1); got {dropout!r}."
            )

        # ── Convolutional feature extractor ──────────────────────────────────
        self.conv_blocks: nn.Sequential = _build_conv_blocks(
            conv_channels, use_batchnorm, activation
        )

        # ── Infer the FC input size via a dummy forward pass ─────────────────
        #
        # Rationale: computing conv_channels[-1] × (28 // 2^N)² manually
        # hard-codes assumptions about padding, stride, and pooling.  A
        # single dummy pass is always correct regardless of architecture
        # changes and costs nothing — torch.no_grad() suppresses all
        # gradient bookkeeping, and this code runs once at construction.
        with torch.no_grad():
            _dummy = torch.zeros(1, 1, 28, 28)
            fc_in  = self.conv_blocks(_dummy).numel()   # C_last × H_out × W_out

        # ── Flatten layer ────────────────────────────────────────────────────
        self.flatten: nn.Flatten = nn.Flatten()

        # ── Fully-connected classification head ──────────────────────────────
        #
        # Two-layer FC head:
        #   fc1      : projects from feature space to a lower-dimensional space
        #   act_fc   : non-linearity
        #   dropout  : regularisation; zero'd during eval()
        #   fc2      : maps to class logits
        #
        # Named OrderedDict gives readable state_dict keys:
        #   fc_head.fc1.weight, fc_head.fc2.weight, …
        # instead of positional fc_head.0.weight, fc_head.2.weight.
        self.fc_head: nn.Sequential = nn.Sequential(OrderedDict([
            ("fc1",     nn.Linear(fc_in, fc_hidden, bias=True)),
            ("act_fc",  _make_activation(activation)),
            ("dropout", nn.Dropout(p=dropout)),
            ("fc2",     nn.Linear(fc_hidden, num_classes, bias=True)),
        ]))

        # ── Store metadata for external introspection ─────────────────────────
        self.conv_channels: list[int] = conv_channels
        self.fc_input_size: int       = fc_in
        self.num_classes:   int       = num_classes

        # ── Weight initialisation ─────────────────────────────────────────────
        _init_weights(self, activation)

        # ── Log summary ──────────────────────────────────────────────────────
        stats = self.count_parameters()
        logger.info(
            "MNISTConvNet ready | blocks=%d  channels=%s  fc_hidden=%d  "
            "fc_in=%d  params=%s  trainable=%s",
            len(conv_channels),
            conv_channels,
            fc_hidden,
            fc_in,
            f"{stats['total']:,}",
            f"{stats['trainable']:,}",
        )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the full forward pass from image to class logits.

        Tensor shapes at each stage (B=batch size, default config):

            Input          (B,  1, 28, 28)

            ─ Conv Block 1 ─────────────────────────────────────────────
            conv1          (B, 32, 28, 28)   3×3 conv, padding=1 preserves H,W
            bn1            (B, 32, 28, 28)   normalises pre-activation values
            act1           (B, 32, 28, 28)   element-wise non-linearity
            pool1          (B, 32, 14, 14)   MaxPool2d halves H and W

            ─ Conv Block 2 ─────────────────────────────────────────────
            conv2          (B, 64, 14, 14)
            bn2            (B, 64, 14, 14)
            act2           (B, 64, 14, 14)
            pool2          (B, 64,  7,  7)   28 → 14 → 7

            ─ FC Head ──────────────────────────────────────────────────
            flatten        (B, 3136)         64 × 7 × 7 = 3136
            fc1            (B,  256)         learned projection
            act_fc         (B,  256)
            dropout        (B,  256)         p=0.5 train / p=0.0 eval
            fc2            (B,   10)         raw logits — no softmax

        Args:
            x: Float tensor, shape (B, 1, 28, 28).

        Returns:
            Logit tensor, shape (B, 10).
            To obtain probabilities: torch.softmax(logits, dim=1).
            Do NOT apply softmax here — CrossEntropyLoss expects raw logits.
        """
        # ── 1. Extract spatial features ───────────────────────────────────────
        x = self.conv_blocks(x)   # (B, 1, 28, 28) → (B, C_last, H_out, W_out)

        # ── 2. Collapse spatial dimensions ────────────────────────────────────
        x = self.flatten(x)        # (B, C_last × H_out × W_out)

        # ── 3. Classify ───────────────────────────────────────────────────────
        x = self.fc_head(x)        # (B, num_classes)

        return x                   # raw logits

    # ── Introspection ─────────────────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """
        Return total and trainable parameter counts.

        Example:
            >>> stats = model.count_parameters()
            >>> print(stats)
            {'total': 1234567, 'trainable': 1234567}
        """
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers (not part of the public API)
# ──────────────────────────────────────────────────────────────────────────────

def _build_conv_blocks(
    conv_channels: list[int],
    use_batchnorm: bool,
    activation: str,
) -> nn.Sequential:
    """
    Build the convolutional feature extractor as a named Sequential.

    Block structure per entry in conv_channels:

        Conv2d(in_ch → out_ch, kernel=3, padding=1, bias=not use_batchnorm)
        BatchNorm2d(out_ch)          ← only when use_batchnorm=True
        Activation                   ← fresh instance per block
        MaxPool2d(kernel=2, stride=2)

    Design notes
    ~~~~~~~~~~~~
    3×3 kernel + padding=1:
        Preserves spatial dimensions through each conv (H_out = H_in,
        W_out = W_in). Downsampling is performed exclusively by MaxPool2d,
        making spatial dimension changes explicit and predictable.

    bias=not use_batchnorm:
        BatchNorm's β parameter subsumes the conv bias — they perform the
        same function (additive shift of pre-activation values).  Keeping
        both wastes parameters without affecting output.

    Conv → BN → Activation order (original BN paper):
        Normalising the pre-activation distribution stabilises training by
        preventing activations from drifting into saturation or zero regions.
        The alternative (BN after activation) normalises a distribution that
        is already non-negative for ReLU, reducing BN's effectiveness.

    MaxPool2d over strided convolution or AvgPool2d:
        Max pooling retains the strongest feature activation in each local
        region — appropriate for detecting presence of a stroke, edge, or
        curve.  Average pooling would dilute strong detections.  Strided
        convolutions are learnable but add parameters and training complexity
        without measurable benefit at this scale.

    Args:
        conv_channels: Output channel count for each block.  The list length
                       determines the number of blocks.
        use_batchnorm: Insert BatchNorm2d after each Conv2d when True.
        activation:    Activation name ('relu' | 'gelu').

    Returns:
        nn.Sequential with named children for readable state_dict keys.
    """
    layers: OrderedDict[str, nn.Module] = OrderedDict()
    in_ch = 1  # MNIST is single-channel greyscale

    for i, out_ch in enumerate(conv_channels, start=1):
        # Convolution ─────────────────────────────────────────────────────────
        layers[f"conv{i}"] = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm,   # redundant when BN follows
        )

        # Batch normalisation ─────────────────────────────────────────────────
        if use_batchnorm:
            layers[f"bn{i}"] = nn.BatchNorm2d(out_ch)

        # Activation ──────────────────────────────────────────────────────────
        layers[f"act{i}"] = _make_activation(activation)

        # Spatial downsampling ─────────────────────────────────────────────────
        # Two blocks on 28×28 input: 28 → 14 → 7.
        # With three blocks: 28 → 14 → 7 → 3 (aggressive but valid).
        layers[f"pool{i}"] = nn.MaxPool2d(kernel_size=2, stride=2)

        in_ch = out_ch

    return nn.Sequential(layers)


def _init_weights(model: nn.Module, activation: str) -> None:
    """
    Apply task-appropriate weight initialisation to every layer.

    Strategy
    ~~~~~~~~
    Conv2d:
        Kaiming Normal, fan_out mode.
        Designed for ReLU-family activations (also conservative for GELU).
        fan_out preserves variance in the backward pass, which benefits
        deeper networks more than fan_in.

    BatchNorm2d:
        weight=1, bias=0 — starts as identity transform, letting the
        network learn useful scale/shift from scratch.

    Linear (fc1, fc2):
        ReLU  → Kaiming Normal (fan_out).
        GELU  → Xavier Normal.
            Xavier assumes symmetric activation with zero mean — GELU is
            smooth and roughly symmetric near zero.  Kaiming assumes a
            half-normal post-activation distribution (ReLU's output is
            non-negative), which over-estimates the variance for GELU.

    All biases:
        Initialised to zero.  The common alternative (uniform small noise)
        offers no empirical benefit for this architecture size.

    Args:
        model:      The module (or any nn.Module) whose sub-modules will
                    be initialised.
        activation: Config activation name; controls Linear init strategy.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",   # conservative default; valid for gelu too
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)   # scale γ = 1
            nn.init.zeros_(module.bias)    # shift β = 0

        elif isinstance(module, nn.Linear):
            if activation == "relu":
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            else:
                # Xavier Normal for symmetric activations (gelu, tanh)
                nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

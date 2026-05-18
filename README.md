# MNIST Handwritten Digit Classification

> Specification-Driven Development (SDD) · PyTorch 2.1+ · CNN ≥ 99 % test accuracy target

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-142%20passed-brightgreen)
![Spec](https://img.shields.io/badge/spec-SDD--MNIST--001%20v2.0.0-informational)

A research-grade, fully reproducible pipeline for MNIST digit classification (0 – 9).
Every parameter is config-driven; every architectural decision is documented in [`CLAUDE.md`](CLAUDE.md);
every requirement has a test. The primary model is a configurable CNN; an MLP baseline follows in Phase 2.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Folder Structure](#3-folder-structure)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Expected Outputs](#8-expected-outputs)
9. [Metrics Reference](#9-metrics-reference)
10. [Visualizations](#10-visualizations)
11. [Running Tests](#11-running-tests)
12. [Reproducibility](#12-reproducibility)
13. [Implementation Status](#13-implementation-status)
14. [References](#14-references)

---

## 1. Project Overview

| Field | Detail |
|---|---|
| **Task** | 10-class digit classification on MNIST (0 – 9) |
| **Primary model** | `MNISTConvNet` — configurable 2-block CNN |
| **Baseline model** | `MNISTNet` — MLP (Phase 2, planned) |
| **Accuracy target** | CNN ≥ 99.0 % · MLP ≥ 97.5 % on the held-out test set |
| **Parameters** | 824,554 (default config; all trainable) |
| **Development method** | Specification-Driven Development (SDD) |
| **Specification** | [`SPEC.md`](SPEC.md) — SDD-MNIST-001 v2.0.0 |
| **Developer notes** | [`CLAUDE.md`](CLAUDE.md) — architectural decisions and rationale |

### What makes this project different from a typical MNIST notebook

- **Spec first** — every module was written against a reviewed requirement. No code precedes its spec section.
- **Single YAML control plane** — every hyperparameter lives in `config/default.yaml`; no magic numbers in source.
- **Self-describing checkpoints** — each `.pt` file embeds the config used to produce it, so evaluation never requires the original config file.
- **Stratified train/val split** — guaranteed class balance, not approximate, using `sklearn.StratifiedShuffleSplit`.
- **Full metric suite** — per-epoch CSV log, per-class precision/recall/F1, confusion matrix, macro F1, all saved automatically.

---

## 2. Architecture

### CNN (primary model)

```
Input              (B,  1, 28, 28)   greyscale MNIST image

── Conv Block 1 ─────────────────────────────────────────────
Conv2d(1 → 32, k=3, pad=1)          (B, 32, 28, 28)
BatchNorm2d(32)                      (B, 32, 28, 28)
ReLU                                 (B, 32, 28, 28)
MaxPool2d(kernel=2, stride=2)        (B, 32, 14, 14)

── Conv Block 2 ─────────────────────────────────────────────
Conv2d(32 → 64, k=3, pad=1)         (B, 64, 14, 14)
BatchNorm2d(64)                      (B, 64, 14, 14)
ReLU                                 (B, 64, 14, 14)
MaxPool2d(kernel=2, stride=2)        (B, 64,  7,  7)

── FC Head ──────────────────────────────────────────────────
Flatten                              (B, 3136)      [64 × 7 × 7]
Linear(3136 → 256)                   (B, 256)
ReLU
Dropout(p=0.5)
Linear(256 → 10)                     (B,  10)       raw logits
```

**Key design choices** (full rationale in [`CLAUDE.md §5`](CLAUDE.md)):

- **3 × 3 kernels, `padding=1`** — preserves spatial dimensions through each conv; downsampling is done exclusively by MaxPool2d, making the spatial flow explicit.
- **BatchNorm before activation** — normalises pre-activation values, stabilising training and preventing saturation.
- **Conv bias disabled when BatchNorm follows** — BatchNorm's β parameter subsumes the bias, eliminating redundant parameters.
- **Dropout only in the FC layer** — spatial dropout on MNIST's small feature maps would destroy correlated stroke features; FC dropout is sufficient regularisation.
- **Raw logits out, no softmax** — `CrossEntropyLoss` applies `log_softmax` internally; applying softmax before the loss degrades numerical precision.

The depth (`conv_channels`) and width (`fc_hidden`) are fully configurable without modifying source code.

---

## 3. Folder Structure

```
mnist-sdd/
│
├── config/
│   └── default.yaml          # single control plane for all hyperparameters [Phase 0]
│
├── scripts/
│   ├── evaluate.py           # ✅ post-training evaluation CLI
│   ├── train.py              # 🔲 training entry point        [Phase 6]
│   ├── infer.py              # 🔲 single-image / batch inference [Phase 5]
│   ├── visualize.py          # 🔲 plot training curves from saved runs [Phase 4]
│   └── tune.py               # 🔲 Optuna hyperparameter search [Phase 7]
│
├── src/
│   ├── utils.py              # ✅ set_seed, AverageMeter, checkpoint I/O, Timer, …
│   │
│   ├── data/
│   │   └── dataset.py        # ✅ MNIST download, stratified split, DataLoaders
│   │
│   ├── model/
│   │   ├── cnn.py            # ✅ MNISTConvNet (primary)
│   │   └── mlp.py            # 🔲 MNISTNet baseline            [Phase 2]
│   │
│   ├── training/
│   │   ├── trainer.py        # ✅ Trainer — fit(), evaluate()
│   │   ├── early_stopping.py # ✅ EarlyStopping
│   │   └── metrics.py        # ✅ accuracy, confusion matrix, macro F1, report
│   │
│   ├── evaluation/
│   │   └── evaluator.py      # ✅ standalone Evaluator — loads checkpoint, runs test set
│   │
│   ├── inference/            # 🔲 Predictor — single image / batch  [Phase 5]
│   ├── visualization/        # 🔲 plot_curves, plot_confusion        [Phase 4]
│   └── tuning/               # 🔲 Optuna Tuner                       [Phase 7]
│
├── tests/
│   ├── conftest.py           # shared fixtures (base_cfg, synthetic_loader)
│   ├── test_cnn.py           # 31 tests — AC-MDL-001, 003, 004
│   ├── test_data.py          # 13 tests — AC-DAT-001 … 005
│   ├── test_trainer.py       # 36 tests — AC-TRN-001 … 005
│   ├── test_evaluator.py     # 19 tests — AC-EVL-001 … 004
│   └── test_utils.py         # 43 tests — all utility functions
│
├── data/                     # MNIST raw files (auto-downloaded, git-ignored)
├── checkpoints/              # saved .pt files        (auto-created, git-ignored)
├── runs/                     # metrics CSV, logs, plots (auto-created, git-ignored)
│
├── requirements.txt
├── requirements-dev.txt
├── SPEC.md                   # SDD-MNIST-001 v2.0.0 — authoritative contract
└── CLAUDE.md                 # developer notes — decisions and rationale
```

> **Legend:** ✅ implemented and tested · 🔲 planned (phase shown)

---

## 4. Installation

### Prerequisites

- Python 3.10 or later
- pip 23+
- (Optional) CUDA-capable GPU — the pipeline auto-detects and uses it if available

### Step 1 — Clone the repository

```bash
git clone https://github.com/gburnwaliisc/mnist-sdd.git
cd mnist-sdd
```

### Step 2 — Create a virtual environment

**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3 — Install runtime dependencies

```bash
pip install -r requirements.txt
```

> **GPU note** — the default `requirements.txt` installs the CPU-only torch wheel.
> For CUDA support, replace the `torch` + `torchvision` lines with the appropriate
> CUDA wheel **before** running the command above:
>
> ```bash
> # Example: CUDA 12.8
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
> pip install -r requirements.txt
> ```
>
> See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the full matrix.

### Step 4 — (Optional) Install development dependencies

Required only if you want to run the test suite or contribute:

```bash
pip install -r requirements-dev.txt
```

### Verify installation

```bash
python - <<'EOF'
import torch, torchvision
from src.model.cnn import MNISTConvNet
print(f"torch      {torch.__version__}")
print(f"torchvision {torchvision.__version__}")
print(f"CUDA       {'available' if torch.cuda.is_available() else 'not available (CPU mode)'}")
cfg = {"model": {"conv_channels": [32, 64], "fc_hidden": 256, "dropout": 0.5,
                  "use_batchnorm": True, "activation": "relu", "num_classes": 10}}
m = MNISTConvNet(cfg)
print(f"Model OK   {m.count_parameters()['total']:,} parameters")
EOF
```

Expected output:
```
torch       2.12.0+cpu
torchvision 0.27.0+cpu
CUDA        not available (CPU mode)
Model OK    824,554 parameters
```

---

## 5. Configuration

All behaviour is controlled through a single YAML file (`config/default.yaml`).
No hyperparameter should ever be hard-coded in source.

```yaml
seed: 42                        # global RNG seed — ensures reproducible splits

data:
  root: ./data                  # MNIST download directory
  batch_size: 64
  num_workers: 0                # set > 0 for multi-process loading (Linux/macOS)
  pin_memory: false             # set true when using a GPU
  val_split: 0.1                # fraction of training set held out for validation
  augmentation: false           # random affine transforms during training
  normalize:
    mean: 0.1307                # MNIST channel mean (fixed constant)
    std:  0.3081                # MNIST channel std  (fixed constant)

model:
  arch: cnn                     # "cnn" | "mlp"
  conv_channels: [32, 64]       # one conv block per entry; depth is configurable
  fc_hidden: 256
  use_batchnorm: true
  activation: relu              # "relu" | "gelu"
  dropout: 0.5
  num_classes: 10

training:
  epochs: 20
  optimizer: adam               # "adam" | "sgd"
  learning_rate: 0.001
  weight_decay: 0.0001
  momentum: 0.9                 # SGD only
  lr_scheduler:
    type: cosine                # "cosine" | "step" | "plateau" | "none"
    T_max: 20                   # cosine: decay period (defaults to epochs)
  early_stopping:
    enabled: true
    monitor: val_loss           # "val_loss" | "val_acc"
    patience: 5
    min_delta: 0.0
  checkpoint:
    dir: ./checkpoints
    save_best_only: true        # best.pt + last.pt always saved regardless
    save_every_epoch: false     # also save epoch_NNN.pt each epoch
  log_dir: ./runs
  auto_visualize: false         # generate training_curves.png after fit()

inference:
  checkpoint: ./checkpoints/best.pt
  device: cpu
```

### Key configuration notes

| Key | Choices | Effect |
|---|---|---|
| `model.conv_channels` | any list, e.g. `[16, 32, 64]` | number of conv blocks equals list length |
| `model.activation` | `relu`, `gelu` | weight initialisation also adapts (Kaiming vs Xavier) |
| `training.optimizer` | `adam`, `sgd` | Adam uses `lr` + `weight_decay`; SGD also uses `momentum` |
| `training.lr_scheduler.type` | `cosine`, `step`, `plateau`, `none` | `ReduceLROnPlateau` requires the metric as an argument; handled automatically |
| `training.early_stopping.monitor` | `val_loss`, `val_acc` | determines improvement direction (↓ for loss, ↑ for acc) |
| `num_workers` | `0` on Windows, `≥1` on Linux/macOS | Windows DataLoader uses `spawn`; > 0 without `__main__` guard causes deadlock |

---

## 6. Training

> **Note:** `scripts/train.py` is planned for Phase 6. The `Trainer` class is fully
> implemented and can be used directly from Python today.

### Programmatic API (available now)

```python
import torch
from src.data.dataset import get_dataloaders
from src.model import build_model
from src.training.trainer import Trainer
from src.utils import set_seed

cfg = {
    "seed": 42,
    "data": {
        "root": "./data", "batch_size": 64, "num_workers": 0,
        "pin_memory": False, "val_split": 0.1, "augmentation": False,
        "normalize": {"mean": 0.1307, "std": 0.3081},
    },
    "model": {
        "arch": "cnn", "conv_channels": [32, 64], "fc_hidden": 256,
        "use_batchnorm": True, "activation": "relu", "dropout": 0.5,
        "num_classes": 10,
    },
    "training": {
        "epochs": 20, "optimizer": "adam", "learning_rate": 1e-3,
        "weight_decay": 1e-4, "momentum": 0.9,
        "lr_scheduler": {"type": "cosine"},
        "early_stopping": {"enabled": True, "monitor": "val_loss",
                           "patience": 5, "min_delta": 0.0},
        "checkpoint": {"dir": "./checkpoints", "save_best_only": True,
                       "save_every_epoch": False},
        "log_dir": "./runs", "auto_visualize": True,
    },
}

set_seed(cfg["seed"])
train_loader, val_loader, test_loader = get_dataloaders(cfg)

model   = build_model(cfg)
trainer = Trainer(model, cfg, run_id="cnn_first_run")

final_metrics = trainer.fit(train_loader, val_loader)
test_results  = trainer.evaluate(test_loader)

print(f"Test accuracy : {test_results['test_acc']:.4f}")
print(f"Macro F1      : {test_results['macro_f1']:.4f}")
```

### CLI (Phase 6 — planned)

```bash
# Standard training run (auto-generates run_id = cnn_YYYYMMDD_HHMMSS)
python scripts/train.py --config config/default.yaml

# Explicit run identifier
python scripts/train.py --config config/default.yaml --run-id cnn_experiment_01

# Resume from last checkpoint
python scripts/train.py --config config/default.yaml \
    --resume checkpoints/cnn_20260518_143201/last.pt
```

### Optimizer and scheduler options

| Config value | Class | Notes |
|---|---|---|
| `optimizer: adam` | `torch.optim.Adam` | default; good for training from scratch |
| `optimizer: sgd` | `torch.optim.SGD` | pairs well with cosine LR; can outperform Adam at high accuracy |
| `lr_scheduler: cosine` | `CosineAnnealingLR` | smooth decay from `lr` to 0 over `T_max` epochs |
| `lr_scheduler: step` | `StepLR` | requires `step_size` + `gamma` sub-keys |
| `lr_scheduler: plateau` | `ReduceLROnPlateau` | reduces LR when `val_loss` plateaus |
| `lr_scheduler: none` | — | constant LR throughout |

---

## 7. Evaluation

### Evaluate a saved checkpoint

```bash
python scripts/evaluate.py --checkpoint checkpoints/cnn_20260518/best.pt
```

The checkpoint is **self-describing** — it embeds the training config, so the correct
model architecture is reconstructed automatically regardless of the current `default.yaml`.

#### Optional flags

| Flag | Description | Default |
|---|---|---|
| `--checkpoint PATH` | Path to the `.pt` checkpoint file | required |
| `--run-dir DIR` | Directory for output files | `runs/{checkpoint_parent}/` |
| `--batch-size N` | Override the checkpoint's batch size | from checkpoint config |
| `--log-level LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO` |

#### Console output example

```
==============================================================
  MNIST Evaluation Report
  Checkpoint : checkpoints/cnn_20260518/best.pt
  Epoch      : 18
==============================================================
  Samples tested :     10,000
  Test loss      :     0.0287
  Test accuracy  :     99.13%
  Macro F1       :     0.9912

  Digit  Precision    Recall       F1       Support
  -----  ---------   ------       --       -------
    0      99.49%    99.59%    99.54%         980
    1      99.65%    99.74%    99.69%        1135
    2      99.22%    98.74%    98.98%        1032
    3      98.82%    99.50%    99.16%        1010
    4      99.39%    99.29%    99.34%         982
    5      98.76%    99.11%    98.93%         892
    6      99.48%    99.38%    99.43%         958
    7      98.93%    99.32%    99.12%        1028
    8      98.96%    98.46%    98.71%         974
    9      99.01%    98.81%    98.91%        1009
  -----  ---------   ------       --       -------
  Macro   99.17%    99.19%    99.18%       10000
==============================================================
  Confusion matrix : runs/cnn_20260518/confusion_matrix.png
  Full results     : runs/cnn_20260518/test_results.json
==============================================================
```

### Inference on new images (Phase 5 — planned)

```bash
# Single image
python scripts/infer.py \
    --image path/to/digit.png \
    --checkpoint checkpoints/cnn_20260518/best.pt

# Directory of images
python scripts/infer.py \
    --image-dir path/to/images/ \
    --checkpoint checkpoints/cnn_20260518/best.pt
```

---

## 8. Expected Outputs

After a successful training run, the following directory structure is created automatically:

```
checkpoints/
└── cnn_20260518_143201/
    ├── best.pt          # checkpoint with the best monitored metric
    ├── last.pt          # checkpoint from the final epoch
    └── epoch_NNN.pt     # per-epoch (only when save_every_epoch: true)

runs/
└── cnn_20260518_143201/
    ├── config.yaml          # full config snapshot — makes the run self-describing
    ├── run.log              # Python logging output (INFO level)
    ├── metrics.csv          # per-epoch training metrics (one row per epoch)
    ├── training_curves.png  # loss + accuracy plots (when auto_visualize: true)
    ├── confusion_matrix.png # raw-count + row-normalised heatmaps
    └── test_results.json    # all final test metrics
```

### `metrics.csv` format

```
epoch,train_loss,val_loss,val_acc,lr
1,0.3142,0.0892,0.9731,0.001000
2,0.0821,0.0613,0.9812,0.000905
...
18,0.0203,0.0287,0.9913,0.000024
```

### `test_results.json` format

```json
{
  "test_loss": 0.0287,
  "test_acc": 0.9913,
  "n_samples": 10000,
  "macro": {
    "f1": 0.9912,
    "precision": 0.9917,
    "recall": 0.9919
  },
  "per_class": [
    {"class": 0, "f1": 0.9954, "precision": 0.9949, "recall": 0.9959, "support": 980},
    ...
  ],
  "confusion_matrix": [[973, 0, 1, ...], ...]
}
```

### Checkpoint schema

Each `.pt` file is a Python dict with the following keys (SPEC §10.4.2):

```python
{
    "epoch":           int,    # epoch number when this checkpoint was saved
    "model_state":     dict,   # model.state_dict()
    "optimizer_state": dict,   # optimizer.state_dict()
    "scheduler_state": dict,   # scheduler.state_dict() or None
    "metrics":         dict,   # per-epoch metrics at this epoch
    "config":          dict,   # full config — checkpoint is self-contained
}
```

---

## 9. Metrics Reference

### Per-epoch metrics (training and validation)

| Symbol | Definition | Range |
|---|---|---|
| `train_loss` | Mean cross-entropy over all training batches (sample-weighted) | [0, ∞) |
| `val_loss` | Mean cross-entropy over all validation batches | [0, ∞) |
| `val_acc` | `correct_predictions / total_val_samples` | [0, 1] |
| `lr` | `optimizer.param_groups[0]['lr']` — current learning rate | (0, ∞) |

Sample-weighted means the final (usually smaller) batch is correctly accounted for.
A naive mean-of-means would underweight it.

### Final test metrics (computed once after training)

| Symbol | Definition |
|---|---|
| `test_loss` | Mean CE loss over the 10,000-sample MNIST test set |
| `test_acc` | `correct / 10000` |
| `per_class_acc[k]` | `correct_for_class_k / total_samples_for_class_k` |
| `confusion_matrix` | 10×10 matrix C where `C[i, j]` = samples with true label i predicted as j |
| `macro_f1` | Unweighted mean of per-class F1 scores |

### Classification report metrics (per class)

All derived from the confusion matrix — no sklearn dependency:

| Symbol | Formula |
|---|---|
| Precision_k | `C[k,k] / C[:,k].sum()` — of all predicted-k, how many were truly k |
| Recall_k | `C[k,k] / C[k,:].sum()` — of all true-k, how many were predicted k |
| F1_k | `2 · P_k · R_k / (P_k + R_k)` — harmonic mean |
| Macro precision | Unweighted mean of Precision_k over all classes |
| Macro recall | Unweighted mean of Recall_k over all classes |
| Macro F1 | Unweighted mean of F1_k over all classes |

Zero-denominator classes (no true or no predicted samples) produce 0.0, not NaN.

### Early stopping

| Parameter | Key | Default |
|---|---|---|
| Monitored metric | `training.early_stopping.monitor` | `val_loss` |
| Patience | `training.early_stopping.patience` | `5` epochs |
| Min improvement | `training.early_stopping.min_delta` | `0.0` |

`val_loss` is the default monitor (preferred over `val_acc`) because loss is a continuous
signal. At high accuracy levels, `val_acc` is nearly quantised to multiples of 0.01 %
and can plateau for many epochs while loss is still improving.

---

## 10. Visualizations

### Training curves (`training_curves.png`)

Generated automatically when `training.auto_visualize: true`, or by calling
`plot_training_curves(epoch_metrics, output_path)` from `src/utils.py`.

Two subplots side by side:
- **Left:** Training loss and validation loss vs epoch (both on the same axes for easy comparison).
- **Right:** Validation accuracy vs epoch (y-axis fixed to [0, 1]).

### Confusion matrix (`confusion_matrix.png`)

Generated by the `Evaluator` after every evaluation run. Two subplots:

- **Left — Raw counts:** `C[i, j]` = number of samples with true digit i predicted as j.
  Useful for finding absolute error counts (e.g., "56 fours were called nines").
- **Right — Row-normalised rates:** each row divided by its true-class total.
  The diagonal equals per-class recall. Normalisation removes the effect of class
  imbalance and makes misclassification rates directly comparable across classes.

Both plots use the `Blues` colormap with per-cell text annotations.

### Planned visualizations (Phase 4)

| Plot | Description |
|---|---|
| Per-class accuracy bar chart | Side-by-side true accuracy for each digit |
| Sample grid | Random sample images with predicted vs true labels |
| Multi-run comparison | Overlay training curves from multiple `run_id`s |
| t-SNE / UMAP embedding | 2-D projection of FC layer activations |

---

## 11. Running Tests

```bash
# Run all 142 tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_cnn.py -v
python -m pytest tests/test_trainer.py -v
python -m pytest tests/test_evaluator.py -v
python -m pytest tests/test_utils.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run a single test by keyword
python -m pytest tests/ -k "test_eval_mode_deterministic"
```

All tests use **synthetic tensors** — no MNIST download is required to run the suite.
A full run completes in roughly 25 seconds on a laptop CPU.

### Test coverage by module

| Test file | Tests | Acceptance criteria covered |
|---|---|---|
| `test_cnn.py` | 31 | AC-MDL-001, AC-MDL-003, AC-MDL-004 |
| `test_data.py` | 13 | AC-DAT-001 … AC-DAT-005 |
| `test_trainer.py` | 36 | AC-TRN-001 … AC-TRN-005 |
| `test_evaluator.py` | 19 | AC-EVL-001 … AC-EVL-004 |
| `test_utils.py` | 43 | all 7 utility sections |
| **Total** | **142** | |

---

## 12. Reproducibility

Full reproducibility requires setting all four independent RNG states before
any data loading or model construction:

```python
from src.utils import set_seed
set_seed(cfg["seed"])   # seeds Python random, NumPy, PyTorch CPU, PyTorch CUDA
                        # also sets cudnn.deterministic=True, benchmark=False
```

Additional mechanisms:

| Source of non-determinism | How it is controlled |
|---|---|
| Train/val split | `StratifiedShuffleSplit(random_state=seed)` |
| DataLoader shuffle | `torch.Generator` seeded with `seed` |
| DataLoader workers | `worker_init_fn` seeds each worker as `seed + worker_id` |
| cuDNN algorithm selection | `benchmark=False` — fixed algorithm per layer |
| Weight initialisation | Seeded before model construction |

> Setting `num_workers > 1` requires the same number of workers across runs.
> Changing `num_workers` after training breaks reproducibility because worker
> ordering is non-deterministic across different worker counts.

---

## 13. Implementation Status

| Phase | Module | Status |
|---|---|---|
| Data pipeline | `src/data/dataset.py` | ✅ Complete |
| CNN model | `src/model/cnn.py` | ✅ Complete |
| Training pipeline | `src/training/` (Trainer, EarlyStopping, metrics) | ✅ Complete |
| Evaluation module | `src/evaluation/evaluator.py` | ✅ Complete |
| Cross-cutting utilities | `src/utils.py` | ✅ Complete |
| Evaluation CLI | `scripts/evaluate.py` | ✅ Complete |
| Config system | `config/default.yaml` + `src/config.py` | 🔲 Phase 0 |
| MLP baseline | `src/model/mlp.py` | 🔲 Phase 2 |
| Visualization scripts | `src/visualization/` | 🔲 Phase 4 |
| Inference pipeline | `src/inference/predictor.py` | 🔲 Phase 5 |
| Training CLI | `scripts/train.py` | 🔲 Phase 6 |
| Hyperparameter tuning | `src/tuning/tuner.py` + `scripts/tune.py` | 🔲 Phase 7 |

The specification for all planned phases is complete in [`SPEC.md`](SPEC.md).

---

## 14. References

| Resource | Purpose |
|---|---|
| [LeCun et al., 1998 — LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | Original MNIST CNN architecture this design is inspired by |
| [Ioffe & Szegedy, 2015 — Batch Normalization](https://arxiv.org/abs/1502.03167) | Rationale for Conv → BN → Activation order |
| [He et al., 2015 — Kaiming Init](https://arxiv.org/abs/1502.01852) | Weight initialisation for ReLU networks |
| [Kingma & Ba, 2015 — Adam](https://arxiv.org/abs/1412.6980) | Default optimizer; `lr=1e-3` is the paper's recommended default |
| [Bergstra & Bengio, 2012 — Random Search](https://www.jmlr.org/papers/v13/bergstra12a.html) | Basis for Optuna's TPE search strategy |
| [PyTorch docs — `torch.utils.data`](https://pytorch.org/docs/stable/data.html) | DataLoader, Generator, worker seeding |
| [torchvision — datasets.MNIST](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST) | Dataset download and loading |
| [Optuna — MedianPruner](https://optuna.readthedocs.io/en/stable/reference/pruners.html) | Early trial pruning during hyperparameter search |

---

<p align="center">
  Built with strict Specification-Driven Development at the Indian Institute of Science<br>
  Spec: <a href="SPEC.md">SDD-MNIST-001 v2.0.0</a> &nbsp;·&nbsp;
  Developer notes: <a href="CLAUDE.md">CLAUDE.md</a>
</p>

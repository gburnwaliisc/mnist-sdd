# Software Specification Document
# MNIST Handwritten Digit Classification System

---

| Field | Value |
|---|---|
| **Document ID** | SDD-MNIST-001 |
| **Version** | 2.0.0 |
| **Status** | Draft — Pending Implementation |
| **Date** | 2026-05-18 |
| **Authors** | Ghanshyam (IISc) |
| **Supersedes** | SDD-MNIST-001 v1.0.0 |
| **Repository** | https://github.com/gburnwaliisc/mnist-sdd |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Definitions and Abbreviations](#2-definitions-and-abbreviations)
3. [System Overview](#3-system-overview)
4. [Functional Requirements](#4-functional-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [Dataset Specification](#6-dataset-specification)
7. [Model Architecture Specification](#7-model-architecture-specification)
8. [Training, Validation and Testing Specification](#8-training-validation-and-testing-specification)
9. [Metrics Specification](#9-metrics-specification)
10. [Logging and Checkpointing Specification](#10-logging-and-checkpointing-specification)
11. [Inference Pipeline Specification](#11-inference-pipeline-specification)
12. [Hyperparameter Tuning Specification](#12-hyperparameter-tuning-specification)
13. [Visualization Specification](#13-visualization-specification)
14. [Configuration Specification](#14-configuration-specification)
15. [Reproducibility Requirements](#15-reproducibility-requirements)
16. [Dependency Requirements](#16-dependency-requirements)
17. [CLI Specification](#17-cli-specification)
18. [Test Specification](#18-test-specification)
19. [Acceptance Criteria](#19-acceptance-criteria)
20. [Project Folder Structure](#20-project-folder-structure)
21. [Out of Scope](#21-out-of-scope)
22. [Change Log](#22-change-log)

---

## 1. Introduction

### 1.1 Purpose

This document defines the complete functional and non-functional requirements for the MNIST Handwritten Digit Classification System. It serves as the authoritative contract between specification and implementation. No module shall be implemented without a corresponding approved specification section in this document. Any deviation from this document during implementation shall require a formal revision of this document prior to code being written.

### 1.2 Scope

The system shall provide:

- A configurable Convolutional Neural Network (CNN) and a Multi-Layer Perceptron (MLP) baseline for classifying greyscale 28×28 images into one of ten digit classes (0–9).
- A data loading pipeline producing reproducible train, validation, and test splits.
- A training and validation pipeline with early stopping, learning-rate scheduling, and checkpointing.
- A held-out test evaluation pipeline producing final performance metrics.
- An inference pipeline supporting single-image, batch, and tensor inputs.
- A hyperparameter tuning pipeline using Optuna.
- A visualization pipeline producing publication-ready plots.
- A single YAML configuration file as the exclusive control plane for all pipelines.

### 1.3 Intended Audience

- Implementation engineer (author)
- Code reviewers
- Future maintainers

### 1.4 Requirement Keywords

The following keywords carry the meanings defined in RFC 2119:

- **shall** — mandatory; non-compliance constitutes a defect.
- **should** — strongly recommended; deviation requires documented justification.
- **may** — permitted but not required.

### 1.5 Implementation Contract

- Every `shall` requirement has a corresponding acceptance criterion in §19.
- Status tags track implementation progress inline:
  `[ ]` not started · `[~]` in progress · `[x]` complete.

---

## 2. Definitions and Abbreviations

| Term | Definition |
|---|---|
| **CNN** | Convolutional Neural Network |
| **MLP** | Multi-Layer Perceptron (fully-connected / dense network) |
| **MNIST** | Modified National Institute of Standards and Technology dataset of handwritten digits |
| **epoch** | One full pass over the training dataset |
| **batch** | A subset of the dataset processed in a single forward/backward pass |
| **logit** | Raw unnormalized score output by the final linear layer before softmax |
| **run** | A single execution of the training pipeline, identified by a unique `run_id` |
| **checkpoint** | A saved file containing model weights and optimizer state at a given epoch |
| **val split** | The fraction of the training set held out for validation during training |
| **BN** | Batch Normalization |
| **LR** | Learning Rate |
| **CE** | Cross-Entropy loss |
| **SDD** | Specification-Driven Development |
| **AC** | Acceptance Criterion |
| **FR** | Functional Requirement |
| **NFR** | Non-Functional Requirement |

---

## 3. System Overview

### 3.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    config/default.yaml                      │
│              (single control plane for all pipelines)       │
└──────────────┬──────────────────────────────────────────────┘
               │ load_config()
               ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                         src/                                        │
 │                                                                     │
 │  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌────────────┐  │
 │  │  data/   │──▶│  training/   │──▶│  model/  │   │ inference/ │  │
 │  │          │   │  trainer.py  │   │  cnn.py  │   │predictor.py│  │
 │  │dataloader│   │  metrics.py  │   │  mlp.py  │   │            │  │
 │  │transforms│   │early_stop.py │   └──────────┘   └────────────┘  │
 │  └──────────┘   └──────────────┘                                   │
 │                        │                                           │
 │  ┌──────────┐          │          ┌───────────────────────────┐   │
 │  │ tuning/  │          ▼          │     visualization/        │   │
 │  │ tuner.py │   ┌────────────┐    │ plot_curves.py            │   │
 │  └──────────┘   │checkpoints/│    │ plot_confusion.py         │   │
 │                 │ runs/      │    │ plot_samples.py           │   │
 │                 └────────────┘    └───────────────────────────┘   │
 └─────────────────────────────────────────────────────────────────────┘
               │
               ▼
 ┌─────────────────────────┐
 │       scripts/          │
 │  train.py evaluate.py   │
 │  infer.py  tune.py      │
 │  visualize.py           │
 └─────────────────────────┘
```

### 3.2 Pipeline Execution Order

```
Data Loading → Model Instantiation → Training Loop → Validation Loop
    → Checkpointing → (repeat per epoch) → Early Stopping
    → Test Evaluation → Visualization → [optional: Inference / Tuning]
```

---

## 4. Functional Requirements

### 4.1 Configuration Management

| ID | Requirement |
|---|---|
| **FR-CFG-001** | The system shall load all pipeline parameters from a single YAML configuration file. |
| **FR-CFG-002** | No source module shall contain hard-coded hyperparameters, file paths, or numeric constants that appear in the configuration file. |
| **FR-CFG-003** | The system shall expose a `load_config(path: str) -> dict` function as the sole config loading interface. |
| **FR-CFG-004** | All CLI entry points shall accept a `--config <path>` argument defaulting to `config/default.yaml`. |
| **FR-CFG-005** | The system shall save a snapshot of the full resolved config to `runs/{run_id}/config.yaml` at the start of every training run. |
| **FR-CFG-006** | Config sections shall be passed to modules as sub-dicts; modules shall not access sibling sections directly. |

### 4.2 Data Pipeline

| ID | Requirement |
|---|---|
| **FR-DAT-001** | The system shall load the MNIST dataset via `torchvision.datasets.MNIST`. |
| **FR-DAT-002** | The system shall automatically download MNIST to the configured `data.root` directory if not already present. |
| **FR-DAT-003** | The system shall produce three non-overlapping splits: train, validation, and test. |
| **FR-DAT-004** | The train/validation split shall be stratified, preserving the class distribution of the original training set in both subsets. |
| **FR-DAT-005** | The test split shall use the official MNIST test partition (10,000 samples) and shall not be used during training or hyperparameter selection. |
| **FR-DAT-006** | All splits shall be reproducible: identical seeds shall yield identical splits. |
| **FR-DAT-007** | The system shall apply pixel normalization using the dataset-level mean and standard deviation from config. |
| **FR-DAT-008** | Optional data augmentation (random affine) shall be configurable and applied to the training split only. |
| **FR-DAT-009** | The data pipeline shall return three `torch.utils.data.DataLoader` objects via a single `get_dataloaders(cfg)` function. |
| **FR-DAT-010** | Image tensors presented to the model shall be of shape `(B, 1, 28, 28)` for the CNN and `(B, 784)` for the MLP. |

### 4.3 Model

| ID | Requirement |
|---|---|
| **FR-MDL-001** | The system shall implement a configurable CNN model (`MNISTConvNet`) as the primary architecture. |
| **FR-MDL-002** | The system shall implement a configurable MLP model (`MNISTNet`) as a baseline architecture. |
| **FR-MDL-003** | The architecture to use shall be selected via the `model.arch` configuration key (`cnn` or `mlp`). |
| **FR-MDL-004** | Both models shall output raw logits of shape `(B, 10)`; no softmax shall be applied inside the model. |
| **FR-MDL-005** | Both models shall support `model.eval()` mode, which disables dropout and uses running statistics in batch normalisation layers. |
| **FR-MDL-006** | All architectural hyperparameters (layer widths, kernel sizes, dropout rate, activation function) shall be read from config. |

### 4.4 Training and Validation

| ID | Requirement |
|---|---|
| **FR-TRN-001** | The system shall train the model using `nn.CrossEntropyLoss`. |
| **FR-TRN-002** | The system shall support Adam and SGD optimizers, selected via config. |
| **FR-TRN-003** | The system shall support StepLR, CosineAnnealingLR, ReduceLROnPlateau, and no-op schedulers, selected via config. |
| **FR-TRN-004** | The system shall run a full validation pass after every training epoch. |
| **FR-TRN-005** | Validation shall be performed with `torch.no_grad()` to prevent gradient accumulation. |
| **FR-TRN-006** | The system shall implement early stopping: training shall halt when the monitored metric (`val_loss` or `val_acc`) fails to improve for `patience` consecutive epochs. |
| **FR-TRN-007** | The system shall support resuming training from a saved checkpoint via `--resume`. |
| **FR-TRN-008** | The training pipeline shall log metrics to CSV and to stdout after every epoch. |

### 4.5 Test Evaluation

| ID | Requirement |
|---|---|
| **FR-TST-001** | The system shall evaluate the best checkpoint on the held-out test set exactly once, after training is complete. |
| **FR-TST-002** | The test evaluation shall compute test loss, test accuracy, per-class accuracy, and the confusion matrix. |
| **FR-TST-003** | Test results shall be written to `runs/{run_id}/test_results.json`. |
| **FR-TST-004** | The test set shall not influence any training decision, hyperparameter selection, or early stopping criterion. |

### 4.6 Inference

| ID | Requirement |
|---|---|
| **FR-INF-001** | The inference pipeline shall support single-image input (file path). |
| **FR-INF-002** | The inference pipeline shall support batch input (directory of images). |
| **FR-INF-003** | The inference pipeline shall support direct tensor input of shape `(B, 1, 28, 28)` or `(B, 784)`. |
| **FR-INF-004** | For file inputs, the pipeline shall resize images to 28×28 and convert to greyscale if necessary. |
| **FR-INF-005** | The pipeline shall apply the same normalization as used during training. |
| **FR-INF-006** | Each prediction shall produce a `Prediction` dataclass containing `label`, `confidence`, and `probabilities`. |
| **FR-INF-007** | The model shall be in `eval()` mode during all inference operations. |

### 4.7 Hyperparameter Tuning

| ID | Requirement |
|---|---|
| **FR-HYP-001** | The system shall use Optuna to conduct hyperparameter optimization. |
| **FR-HYP-002** | The search space shall be defined entirely in the configuration file. |
| **FR-HYP-003** | All trials shall use the same train/validation split to ensure comparability. |
| **FR-HYP-004** | The objective metric shall be `val_acc` (maximize). |
| **FR-HYP-005** | The system shall support Optuna's `MedianPruner` to halt unpromising trials early. |
| **FR-HYP-006** | On completion, the system shall write the best configuration to `runs/tuning/best_config.yaml`. |

### 4.8 Visualization

| ID | Requirement |
|---|---|
| **FR-VIZ-001** | The system shall produce a training/validation loss curve plot. |
| **FR-VIZ-002** | The system shall produce a validation accuracy curve plot with the best-epoch marked. |
| **FR-VIZ-003** | The system shall produce a 10×10 confusion matrix heatmap normalized by true class. |
| **FR-VIZ-004** | The system shall produce a sample prediction grid (5×10) showing one row per class. |
| **FR-VIZ-005** | The system shall produce a misclassified examples plot showing up to 50 test-set errors. |
| **FR-VIZ-006** | The system shall produce a per-class accuracy bar chart. |
| **FR-VIZ-007** | All plots shall be saved as PNG to `runs/{run_id}/plots/` and shall not require a display (use `matplotlib` non-interactive backend). |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| ID | Requirement |
|---|---|
| **NFR-PRF-001** | The CNN shall achieve ≥ 99.0% accuracy on the official MNIST test set within 20 epochs using the default configuration. |
| **NFR-PRF-002** | The MLP baseline shall achieve ≥ 97.5% accuracy on the official MNIST test set within 20 epochs using the default configuration. |
| **NFR-PRF-003** | A single training epoch over the full 60,000-sample training set shall complete in ≤ 60 seconds on a modern CPU. |
| **NFR-PRF-004** | Inference on a single 28×28 image shall complete in ≤ 100 ms on CPU. |

### 5.2 Reproducibility

| ID | Requirement |
|---|---|
| **NFR-REP-001** | Two training runs launched with identical configuration and seed shall produce numerically identical model weights, metrics, and checkpoints. |
| **NFR-REP-002** | The system shall seed Python, NumPy, and PyTorch (CPU and CUDA) from `config.seed` at the start of every run. |
| **NFR-REP-003** | DataLoader worker seeds shall be set via `worker_init_fn` derived from `config.seed`. |

### 5.3 Maintainability

| ID | Requirement |
|---|---|
| **NFR-MNT-001** | Each source module shall have a corresponding unit test file in `tests/`. |
| **NFR-MNT-002** | Public functions and class methods shall carry Python type annotations. |
| **NFR-MNT-003** | No module shall import from a sibling pipeline module (e.g., `training/` must not import from `data/`); coupling passes through the config dict and explicit arguments only. |
| **NFR-MNT-004** | No global mutable state shall exist outside of the config dict. |

### 5.4 Portability

| ID | Requirement |
|---|---|
| **NFR-PRT-001** | The system shall run on Linux, macOS, and Windows without modification. |
| **NFR-PRT-002** | The system shall support CPU-only execution; GPU (CUDA) shall be supported when available and selected via config `inference.device: auto`. |
| **NFR-PRT-003** | All file paths shall use `pathlib.Path` to ensure cross-platform compatibility. |

### 5.5 Usability

| ID | Requirement |
|---|---|
| **NFR-USB-001** | All CLI scripts shall exit with code `0` on success and a non-zero code on any error. |
| **NFR-USB-002** | All CLI scripts shall print a usage message when invoked with `--help`. |
| **NFR-USB-003** | Error messages shall be informative, identifying the failing component and the config key involved where applicable. |

---

## 6. Dataset Specification

### 6.1 Source Dataset

| Property | Value |
|---|---|
| **Name** | MNIST Handwritten Digits |
| **Origin** | Yann LeCun, Corinna Cortes, Christopher Burges |
| **Loader** | `torchvision.datasets.MNIST` |
| **Download target** | `config.data.root` (default: `./data`) |
| **Licence** | Creative Commons Attribution-Share Alike 3.0 |

### 6.2 Dataset Statistics

| Property | Value |
|---|---|
| Image dimensions | 28 × 28 pixels, single channel (greyscale) |
| Pixel value range (raw) | 0–255 (uint8) |
| Pixel value range (after ToTensor) | 0.0–1.0 (float32) |
| Classes | 10 (digits 0–9) |
| Official training samples | 60,000 |
| Official test samples | 10,000 |
| Class distribution (training) | Approximately balanced; ~6,000 per class |

### 6.3 Split Specification

| Split | Source partition | Nominal size | Shuffle | Stratified |
|---|---|---|---|---|
| **train** | MNIST train | `⌊(1 − val_split) × 60000⌋` | Yes | Yes |
| **validation** | MNIST train | `⌈val_split × 60000⌉` | No | Yes |
| **test** | MNIST test | 10,000 (fixed) | No | N/A |

Default `val_split = 0.1` → train: 54,000 · validation: 6,000 · test: 10,000.

### 6.4 Preprocessing Pipeline

Transforms are applied in the following order:

| Step | Train | Validation | Test | Implementation |
|---|---|---|---|---|
| `ToTensor()` | ✓ | ✓ | ✓ | `transforms.ToTensor()` |
| `Normalize(mean, std)` | ✓ | ✓ | ✓ | `transforms.Normalize([0.1307], [0.3081])` |
| `RandomAffine(degrees=10, translate=(0.1,0.1))` | optional | ✗ | ✗ | Enabled via `data.augmentation: true` |

**Normalization values** are the channel-wise mean and standard deviation computed over the full 60,000-sample MNIST training set. These values shall be treated as constants and shall not be recomputed at runtime.

### 6.5 DataLoader Configuration

| Parameter | Config key | Default |
|---|---|---|
| Batch size | `data.batch_size` | 64 |
| Worker processes | `data.num_workers` | 2 |
| Pin memory | `data.pin_memory` | true |
| Shuffle (train only) | — | true |

---

## 7. Model Architecture Specification

### 7.1 Architecture Selection

The architecture to instantiate shall be determined by `model.arch`:

| Value | Class | Module |
|---|---|---|
| `cnn` | `MNISTConvNet` | `src/model/cnn.py` |
| `mlp` | `MNISTNet` | `src/model/mlp.py` |

Both models shall share the same `Trainer`, `Predictor`, and evaluation code.

---

### 7.2 CNN Architecture (`MNISTConvNet`)

#### 7.2.1 Default Layer Blueprint

```
Input:  (B, 1, 28, 28)

── Conv Block 1 ───────────────────────────────────────────
  Conv2d(in=1,  out=conv_channels[0], kernel=3, padding=1)
  BatchNorm2d(conv_channels[0])          [if use_batchnorm]
  Activation                             [relu | gelu]
  MaxPool2d(kernel=2, stride=2)
  → (B, conv_channels[0], 14, 14)

── Conv Block 2 ───────────────────────────────────────────
  Conv2d(in=conv_channels[0], out=conv_channels[1], kernel=3, padding=1)
  BatchNorm2d(conv_channels[1])          [if use_batchnorm]
  Activation
  MaxPool2d(kernel=2, stride=2)
  → (B, conv_channels[1], 7, 7)

── Flatten ────────────────────────────────────────────────
  → (B, conv_channels[1] × 7 × 7)

── FC Block ───────────────────────────────────────────────
  Linear(conv_channels[1] × 7 × 7, fc_hidden)
  Activation
  Dropout(p=dropout)
  Linear(fc_hidden, 10)
  → (B, 10)  [raw logits]
```

#### 7.2.2 CNN Configurable Parameters

| Config key | Type | Default | Description |
|---|---|---|---|
| `model.conv_channels` | list[int] | `[32, 64]` | Output channels for each conv block |
| `model.fc_hidden` | int | `256` | Hidden units in the FC layer |
| `model.dropout` | float | `0.5` | Dropout probability after FC hidden |
| `model.activation` | str | `relu` | `relu` or `gelu` |
| `model.use_batchnorm` | bool | `true` | Whether to apply BN in conv blocks |
| `model.num_classes` | int | `10` | Number of output classes (fixed) |

#### 7.2.3 CNN Public Interface

```python
class MNISTConvNet(nn.Module):
    def __init__(self, cfg: dict) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28) → logits: (B, 10)"""
```

---

### 7.3 MLP Architecture (`MNISTNet`)

#### 7.3.1 Default Layer Blueprint

```
Input:  (B, 784)   [28×28 flattened]

── Hidden Blocks (one per entry in hidden_layers) ─────────
  Linear(in, hidden_layers[i])
  BatchNorm1d(hidden_layers[i])
  Activation                             [relu | tanh]
  Dropout(p=dropout)

── Output ─────────────────────────────────────────────────
  Linear(hidden_layers[-1], 10)
  → (B, 10)  [raw logits]
```

#### 7.3.2 MLP Configurable Parameters

| Config key | Type | Default | Description |
|---|---|---|---|
| `model.input_size` | int | `784` | Flattened input dimension |
| `model.hidden_layers` | list[int] | `[256, 128]` | Width of each hidden layer |
| `model.dropout` | float | `0.2` | Dropout probability per hidden layer |
| `model.activation` | str | `relu` | `relu` or `tanh` |
| `model.num_classes` | int | `10` | Number of output classes |

#### 7.3.3 MLP Public Interface

```python
class MNISTNet(nn.Module):
    def __init__(self, cfg: dict) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 784) → logits: (B, 10)"""
```

---

## 8. Training, Validation and Testing Specification

### 8.1 Training Loop

One epoch shall consist of one complete pass over the training `DataLoader`.

```
for batch (images, labels) in train_loader:
    optimizer.zero_grad()
    logits = model(images)
    loss   = CrossEntropyLoss(logits, labels)
    loss.backward()
    optimizer.step()
    accumulate → train_loss, train_correct
```

### 8.2 Validation Loop

A validation pass shall be executed after every training epoch using `torch.no_grad()`.

```
for batch (images, labels) in val_loader:
    logits = model(images)
    loss   = CrossEntropyLoss(logits, labels)
    accumulate → val_loss, val_correct
```

### 8.3 Optimizer Specification

| Config value | Class | Required config keys |
|---|---|---|
| `adam` | `torch.optim.Adam` | `learning_rate`, `weight_decay` |
| `sgd` | `torch.optim.SGD` | `learning_rate`, `weight_decay`, `momentum` |

Default optimizer: `adam`.

### 8.4 Learning Rate Scheduler Specification

| Config value | Class | Additional keys |
|---|---|---|
| `step` | `StepLR` | `lr_scheduler.step_size`, `lr_scheduler.gamma` |
| `cosine` | `CosineAnnealingLR` | `lr_scheduler.T_max` (defaults to `epochs`) |
| `plateau` | `ReduceLROnPlateau` | `lr_scheduler.factor`, `lr_scheduler.patience` |
| `none` | — | — |

The scheduler shall step once per epoch after the validation loop completes.

### 8.5 Early Stopping Specification

| Parameter | Config key | Default |
|---|---|---|
| Enable | `training.early_stopping.enabled` | `true` |
| Monitored metric | `training.early_stopping.monitor` | `val_loss` |
| Patience (epochs) | `training.early_stopping.patience` | `5` |
| Minimum delta | `training.early_stopping.min_delta` | `0.0` |

Improvement direction:
- `val_loss` → improvement means **decrease** by more than `min_delta`.
- `val_acc` → improvement means **increase** by more than `min_delta`.

When patience is exhausted, training shall halt and the best checkpoint shall be restored before test evaluation.

### 8.6 Test Evaluation

Test evaluation shall load the `best.pt` checkpoint and evaluate on the held-out test `DataLoader` with `torch.no_grad()`. It shall compute all metrics defined in §9. Test evaluation shall be triggered automatically at the end of training and may also be run independently via `scripts/evaluate.py`.

### 8.7 Trainer Public Interface

```python
class Trainer:
    def __init__(self, model: nn.Module, cfg: dict, run_id: str) -> None: ...

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict:
        """
        Run the full training loop. Return a dict of final epoch metrics.
        Saves checkpoints and logs metrics each epoch.
        """

    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Evaluate best checkpoint on test_loader.
        Return dict containing all test metrics.
        """
```

---

## 9. Metrics Specification

### 9.1 Per-Epoch Metrics (train and validation)

| Metric | Symbol | Definition | Range |
|---|---|---|---|
| Training loss | `train_loss` | Mean CE loss over all training batches in the epoch | [0, ∞) |
| Validation loss | `val_loss` | Mean CE loss over all validation batches | [0, ∞) |
| Validation accuracy | `val_acc` | `correct_predictions / total_validation_samples` | [0, 1] |
| Learning rate | `lr` | Current LR returned by `optimizer.param_groups[0]['lr']` | (0, ∞) |

### 9.2 Final Test Metrics (computed once)

| Metric | Symbol | Definition |
|---|---|---|
| Test loss | `test_loss` | Mean CE loss over the 10,000-sample test set |
| Test accuracy | `test_acc` | `correct / 10000` |
| Per-class accuracy | `per_class_acc[k]` | `correct_for_class_k / total_samples_for_class_k`, for k ∈ {0,…,9} |
| Confusion matrix | `confusion_matrix` | 10×10 matrix C where `C[i,j]` = samples with true label i predicted as j |
| Macro F1 | `macro_f1` | Unweighted mean of per-class F1 scores |

### 9.3 Model Summary Metrics (logged once at run start)

| Metric | Description |
|---|---|
| `total_params` | Total number of model parameters |
| `trainable_params` | Number of parameters with `requires_grad=True` |

---

## 10. Logging and Checkpointing Specification

### 10.1 Logging Sinks

| Sink | Content | Frequency |
|---|---|---|
| `stdout` | Epoch, train_loss, val_loss, val_acc, lr | Every epoch |
| `runs/{run_id}/metrics.csv` | All per-epoch metrics (§9.1) | Every epoch (appended) |
| `runs/{run_id}/config.yaml` | Full resolved config snapshot | Once at run start |
| `runs/{run_id}/test_results.json` | All final test metrics (§9.2) | Once after test eval |
| `runs/{run_id}/run.log` | Python `logging` output (INFO level) | Continuous |

### 10.2 Run Identifier

`run_id` shall be auto-generated as `{model.arch}_{YYYYMMDD_HHMMSS}` (e.g., `cnn_20260518_143201`) unless overridden by `--run-id`.

### 10.3 CSV Format

`metrics.csv` shall contain the following columns in order:

```
epoch, train_loss, val_loss, val_acc, lr
```

The header row shall always be written on the first epoch.

### 10.4 Checkpoint Specification

#### 10.4.1 Saved Files

| File | When saved | Config key |
|---|---|---|
| `checkpoints/{run_id}/best.pt` | When monitored metric improves | always |
| `checkpoints/{run_id}/last.pt` | End of every epoch | always |
| `checkpoints/{run_id}/epoch_{n:03d}.pt` | Every epoch | `training.checkpoint.save_every_epoch: true` |

#### 10.4.2 Checkpoint Schema

```python
{
    'epoch':           int,
    'model_state':     dict,   # model.state_dict()
    'optimizer_state': dict,   # optimizer.state_dict()
    'scheduler_state': dict,   # scheduler.state_dict() or None
    'metrics':         dict,   # snapshot of per-epoch metrics at this epoch
    'config':          dict,   # full config — checkpoint is self-contained
}
```

#### 10.4.3 Resume Contract

When `--resume <checkpoint_path>` is supplied:
- `model_state`, `optimizer_state`, and `scheduler_state` shall be restored.
- Training shall continue from `epoch + 1`.
- The same `run_id` shall be reused; `metrics.csv` shall be appended, not overwritten.

---

## 11. Inference Pipeline Specification

### 11.1 Input Modes

| Mode | Method | Input type |
|---|---|---|
| Single image | `predict_image(path)` | File path (PNG/JPG) |
| Image directory | `predict_batch(dir_path)` | Directory path |
| Tensor | `predict_tensor(x)` | `torch.Tensor` shape `(B,1,28,28)` or `(B,784)` |

### 11.2 Preprocessing (file inputs only)

1. Open image with PIL.
2. Convert to greyscale (`L` mode) if not already.
3. Resize to 28×28 using bilinear interpolation.
4. Apply `ToTensor()` and `Normalize(mean, std)` using values from config.

### 11.3 Prediction Output

```python
@dataclass
class Prediction:
    label:         int          # argmax of probabilities
    confidence:    float        # max(probabilities)
    probabilities: list[float]  # softmax(logits), length 10, sums to 1.0
```

Invariant: `label == probabilities.index(confidence)` shall always hold.

### 11.4 Predictor Public Interface

```python
class Predictor:
    def __init__(self, cfg: dict) -> None:
        """Load model from cfg['inference']['checkpoint']. Set eval() mode."""

    def predict_image(self, image_path: str) -> Prediction: ...
    def predict_batch(self, image_dir: str) -> list[Prediction]: ...
    def predict_tensor(self, x: torch.Tensor) -> list[Prediction]: ...
```

---

## 12. Hyperparameter Tuning Specification

### 12.1 Framework

Optuna version ≥ 3.0. Study direction: `maximize` (`val_acc`).

### 12.2 Search Space Schema

Defined under `tuning.search_space` in config. Supported parameter types:

| Type key | Optuna API |
|---|---|
| `float` | `trial.suggest_float(name, low, high)` |
| `float_log` | `trial.suggest_float(name, low, high, log=True)` |
| `int` | `trial.suggest_int(name, low, high)` |
| `categorical` | `trial.suggest_categorical(name, choices)` |

### 12.3 Default Search Space

| Parameter | Type | Range / Choices |
|---|---|---|
| `learning_rate` | `float_log` | `[1e-4, 1e-2]` |
| `batch_size` | `categorical` | `[32, 64, 128]` |
| `conv_channels` | `categorical` | `[[16,32], [32,64], [64,128]]` |
| `fc_hidden` | `categorical` | `[128, 256, 512]` |
| `dropout` | `float` | `[0.0, 0.5]` |
| `optimizer` | `categorical` | `[adam, sgd]` |
| `lr_scheduler` | `categorical` | `[step, cosine, none]` |

### 12.4 Objective Function Contract

For each Optuna trial:
1. Build a fresh model and trainer from the base config merged with trial parameters.
2. Use the fixed-seed train/validation split.
3. Train for `training.epochs` epochs or until early stopping fires.
4. Report `val_acc` of the best epoch to Optuna.
5. Use `trial.report()` and `trial.should_prune()` for `MedianPruner` integration.

### 12.5 Tuner Public Interface

```python
class HyperparamTuner:
    def __init__(self, base_cfg: dict) -> None: ...
    def run(self) -> dict:
        """
        Execute Optuna study for cfg['tuning']['n_trials'] trials.
        Return best hyperparameters as a flat dict.
        """
```

### 12.6 Outputs

| File | Content |
|---|---|
| `runs/tuning/study.pkl` | Serialised Optuna `Study` object |
| `runs/tuning/best_config.yaml` | Base config merged with best trial parameters |
| `runs/tuning/trials.csv` | Per-trial summary: trial number, params, val_acc, duration |

---

## 13. Visualization Specification

All plots shall use `matplotlib` with the non-interactive `Agg` backend. Plots shall be generated by `scripts/visualize.py --run-id <id>` or automatically at the end of training when `training.auto_visualize: true`.

| Plot | Filename | Description |
|---|---|---|
| Loss curves | `loss_curves.png` | `train_loss` and `val_loss` vs epoch; best epoch marked with a vertical dashed line |
| Accuracy curve | `accuracy_curve.png` | `val_acc` vs epoch; best epoch marked; y-axis from 0 to 1 |
| Confusion matrix | `confusion_matrix.png` | 10×10 heatmap; rows = true labels, columns = predicted; normalized by row (true class count); values displayed in cells |
| Sample predictions | `sample_predictions.png` | 5×10 grid; one row per sampled digit class; each cell shows the image, true label, predicted label, and confidence |
| Misclassified examples | `misclassified.png` | Up to 50 test-set errors; each cell shows image, true label, predicted label |
| Per-class accuracy | `per_class_acc.png` | Horizontal bar chart; one bar per digit class; overall test accuracy shown as a vertical reference line |

---

## 14. Configuration Specification

### 14.1 Full Config Schema

```yaml
# ── Global ────────────────────────────────────────────────────────────────────
seed: 42

# ── Data ──────────────────────────────────────────────────────────────────────
data:
  root: ./data
  batch_size: 64
  num_workers: 2
  pin_memory: true
  val_split: 0.1
  augmentation: false
  normalize:
    mean: 0.1307
    std:  0.3081

# ── Model ─────────────────────────────────────────────────────────────────────
model:
  arch: cnn                  # cnn | mlp

  # CNN-specific
  conv_channels: [32, 64]
  fc_hidden: 256
  use_batchnorm: true

  # MLP-specific
  input_size: 784
  hidden_layers: [256, 128]

  # Shared
  num_classes: 10
  activation: relu           # relu | gelu | tanh
  dropout: 0.5

# ── Training ──────────────────────────────────────────────────────────────────
training:
  epochs: 20
  optimizer: adam            # adam | sgd
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  momentum: 0.9              # sgd only

  lr_scheduler:
    type: cosine             # step | cosine | plateau | none
    step_size: 5             # step only
    gamma: 0.5               # step only
    T_max: null              # cosine only; null defaults to epochs
    factor: 0.5              # plateau only
    patience: 3              # plateau only

  early_stopping:
    enabled: true
    monitor: val_loss        # val_loss | val_acc
    patience: 5
    min_delta: 0.0

  checkpoint:
    dir: ./checkpoints
    save_best_only: true
    save_every_epoch: false

  log_dir: ./runs
  auto_visualize: true

# ── Inference ─────────────────────────────────────────────────────────────────
inference:
  checkpoint: ./checkpoints/best.pt
  device: auto               # cpu | cuda | auto

# ── Hyperparameter Tuning ─────────────────────────────────────────────────────
tuning:
  n_trials: 50
  direction: maximize
  pruner: median             # median | none
  storage: null              # Optuna DB URI or null for in-memory

  search_space:
    learning_rate:
      type: float_log
      low: 1.0e-4
      high: 1.0e-2
    batch_size:
      type: categorical
      choices: [32, 64, 128]
    conv_channels:
      type: categorical
      choices:
        - [16, 32]
        - [32, 64]
        - [64, 128]
    fc_hidden:
      type: categorical
      choices: [128, 256, 512]
    dropout:
      type: float
      low: 0.0
      high: 0.5
    optimizer:
      type: categorical
      choices: [adam, sgd]
    lr_scheduler:
      type: categorical
      choices: [step, cosine, none]
```

---

## 15. Reproducibility Requirements

| ID | Requirement | Implementation |
|---|---|---|
| **RR-001** | Python built-in RNG shall be seeded from config | `random.seed(cfg['seed'])` |
| **RR-002** | NumPy RNG shall be seeded from config | `np.random.seed(cfg['seed'])` |
| **RR-003** | PyTorch CPU RNG shall be seeded from config | `torch.manual_seed(cfg['seed'])` |
| **RR-004** | PyTorch CUDA RNG shall be seeded from config | `torch.cuda.manual_seed_all(cfg['seed'])` |
| **RR-005** | cuDNN shall operate in deterministic mode | `torch.backends.cudnn.deterministic = True` |
| **RR-006** | cuDNN benchmark mode shall be disabled | `torch.backends.cudnn.benchmark = False` |
| **RR-007** | DataLoader worker seeds shall be deterministic | `worker_init_fn=lambda id: np.random.seed(cfg['seed'] + id)` |
| **RR-008** | The full resolved config shall be saved to `runs/{run_id}/config.yaml` before any training step executes | `src/config.py` |
| **RR-009** | Two runs with identical config and seed shall produce bit-identical `best.pt` weights on CPU | Verified in `tests/test_trainer.py` |

---

## 16. Dependency Requirements

### 16.1 Runtime Dependencies

| Package | Minimum version | Purpose |
|---|---|---|
| `torch` | 2.1.0 | Deep learning framework |
| `torchvision` | 0.16.0 | MNIST dataset and transforms |
| `numpy` | 1.24.0 | Numerical operations |
| `pyyaml` | 6.0 | Config file parsing |
| `optuna` | 3.0.0 | Hyperparameter optimization |
| `matplotlib` | 3.7.0 | Visualization |
| `pillow` | 9.0.0 | Image loading and resizing in inference |
| `scikit-learn` | 1.3.0 | Stratified split (`StratifiedShuffleSplit`) |
| `pandas` | 2.0.0 | CSV metrics writing |

### 16.2 Development and Test Dependencies

| Package | Minimum version | Purpose |
|---|---|---|
| `pytest` | 7.0.0 | Test runner |
| `pytest-cov` | 4.0.0 | Coverage reporting |

### 16.3 `requirements.txt` Contract

A `requirements.txt` file shall be maintained at the repository root and shall pin exact versions using `==` for all runtime dependencies, derived from a resolved virtual environment. A separate `requirements-dev.txt` shall cover development and test dependencies.

### 16.4 Python Version

The system shall support Python 3.10 and above. Python 3.10 is the minimum due to use of the `match` statement and `tuple[...]` type hint syntax.

---

## 17. CLI Specification

### 17.1 `scripts/train.py`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | str | `config/default.yaml` | Path to YAML config |
| `--run-id` | str | auto-generated | Override the run identifier |
| `--resume` | str | None | Path to checkpoint to resume from |

**Behaviour**: loads config → seeds RNGs → gets dataloaders → instantiates model → runs `Trainer.fit()` → runs `Trainer.evaluate()` → generates visualizations if `auto_visualize` is true.

### 17.2 `scripts/evaluate.py`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | str | `config/default.yaml` | Path to YAML config |
| `--checkpoint` | str | required | Path to `.pt` checkpoint |

**Behaviour**: loads checkpoint → evaluates on test set → prints and saves `test_results.json`.

### 17.3 `scripts/infer.py`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | str | `config/default.yaml` | Path to YAML config |
| `--checkpoint` | str | `inference.checkpoint` | Override checkpoint path |
| `--image` | str | — | Path to a single image file |
| `--image-dir` | str | — | Path to directory of images |

Exactly one of `--image` or `--image-dir` shall be required.

### 17.4 `scripts/tune.py`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | str | `config/default.yaml` | Path to YAML config |

### 17.5 `scripts/visualize.py`

| Flag | Type | Default | Description |
|---|---|---|---|
| `--config` | str | `config/default.yaml` | Path to YAML config |
| `--run-id` | str | required | Run to generate plots for |

---

## 18. Test Specification

### 18.1 General Rules

- Tests shall not download real MNIST data. Synthetic tensors or in-memory fixtures shall be used.
- Tests shall be runnable offline and in CI without GPU.
- Test execution command: `pytest tests/ -v --cov=src`.
- Minimum line coverage target: **80%**.

### 18.2 Test Files

| File | Module under test | Key cases |
|---|---|---|
| `tests/test_config.py` | `src/config.py` | Valid YAML load; missing key raises; default values |
| `tests/test_data.py` | `src/data/dataloader.py` | Loader shapes; split sizes; stratification; reproducibility |
| `tests/test_cnn.py` | `src/model/cnn.py` | Forward shape `(4,10)`; layer count; eval disables dropout |
| `tests/test_mlp.py` | `src/model/mlp.py` | Forward shape `(4,10)`; hidden layer count; eval disables dropout |
| `tests/test_trainer.py` | `src/training/trainer.py` | One-epoch loop completes; metrics.csv written; best.pt saved; early stopping triggers |
| `tests/test_metrics.py` | `src/training/metrics.py` | Accuracy value; confusion matrix shape and sum; F1 sanity |
| `tests/test_predictor.py` | `src/inference/predictor.py` | Tensor prediction shape and type; label == argmax(probs); eval mode |
| `tests/test_visualization.py` | `src/visualization/` | PNG files created; no display required |

---

## 19. Acceptance Criteria

Each criterion maps to one or more functional requirements. All criteria shall pass before a pipeline section is marked `[x]` complete.

### AC-CFG — Configuration

| ID | Criterion |
|---|---|
| **AC-CFG-001** | `load_config('config/default.yaml')` returns a `dict` without raising. |
| **AC-CFG-002** | `load_config` raises `FileNotFoundError` for a missing path. |
| **AC-CFG-003** | After training, `runs/{run_id}/config.yaml` exists and is loadable as a `dict` equal to the original config. |

### AC-DAT — Data Pipeline

| ID | Criterion |
|---|---|
| **AC-DAT-001** | `get_dataloaders(cfg)` returns three `DataLoader` objects without error. |
| **AC-DAT-002** | A batch from `train_loader` has shape `(B, 1, 28, 28)` (CNN) or `(B, 784)` (MLP) and `labels.shape == (B,)`. |
| **AC-DAT-003** | `len(train_loader.dataset) + len(val_loader.dataset) == 60000`. |
| **AC-DAT-004** | The per-class fraction in `val_loader.dataset` is within ±0.5% of the per-class fraction in the original 60,000-sample set. |
| **AC-DAT-005** | Two calls to `get_dataloaders` with the same seed return identical index orderings. |

### AC-MDL — Model

| ID | Criterion |
|---|---|
| **AC-MDL-001** | `MNISTConvNet(cfg)(torch.randn(4,1,28,28)).shape == (4, 10)`. |
| **AC-MDL-002** | `MNISTNet(cfg)(torch.randn(4,784)).shape == (4, 10)`. |
| **AC-MDL-003** | After `model.eval()`, passing the same input twice returns identical outputs (dropout inactive). |
| **AC-MDL-004** | `len(cfg['model']['conv_channels'])` conv blocks exist in `MNISTConvNet`. |
| **AC-MDL-005** | `len(cfg['model']['hidden_layers'])` linear blocks exist in `MNISTNet`. |

### AC-TRN — Training and Validation

| ID | Criterion |
|---|---|
| **AC-TRN-001** | `Trainer.fit()` completes one epoch without raising an exception. |
| **AC-TRN-002** | `runs/{run_id}/metrics.csv` exists after training with columns `epoch,train_loss,val_loss,val_acc,lr`. |
| **AC-TRN-003** | `checkpoints/{run_id}/best.pt` is loadable and `model.load_state_dict(ckpt['model_state'])` succeeds. |
| **AC-TRN-004** | Training halts before `training.epochs` when early stopping patience is exhausted. |
| **AC-TRN-005** | Resuming from `last.pt` and training to the same final epoch yields identical `val_acc` (given same seed). |
| **AC-TRN-006** | The CNN achieves ≥ 99.0% test accuracy within 20 epochs with default config. |
| **AC-TRN-007** | The MLP achieves ≥ 97.5% test accuracy within 20 epochs with default config. |

### AC-TST — Test Evaluation

| ID | Criterion |
|---|---|
| **AC-TST-001** | `runs/{run_id}/test_results.json` exists after training and contains keys `test_loss`, `test_acc`, `per_class_acc`, `confusion_matrix`, `macro_f1`. |
| **AC-TST-002** | `sum(confusion_matrix[i])` equals the number of test samples with true label `i` for all `i`. |

### AC-INF — Inference

| ID | Criterion |
|---|---|
| **AC-INF-001** | `predictor.predict_tensor(torch.zeros(1, 784))` returns a `list[Prediction]` of length 1 without error. |
| **AC-INF-002** | `prediction.label == prediction.probabilities.index(prediction.confidence)` for every returned `Prediction`. |
| **AC-INF-003** | `abs(sum(prediction.probabilities) - 1.0) < 1e-5` for every returned `Prediction`. |
| **AC-INF-004** | The model is in `eval()` mode during inference: `model.training == False`. |

### AC-HYP — Hyperparameter Tuning

| ID | Criterion |
|---|---|
| **AC-HYP-001** | A study with `n_trials=2` completes without error. |
| **AC-HYP-002** | `runs/tuning/best_config.yaml` is valid YAML and passes `load_config`. |
| **AC-HYP-003** | `runs/tuning/trials.csv` exists with at least one row per completed trial. |

### AC-VIZ — Visualization

| ID | Criterion |
|---|---|
| **AC-VIZ-001** | All six PNG files defined in §13 exist in `runs/{run_id}/plots/` after `scripts/visualize.py` runs. |
| **AC-VIZ-002** | Visualization completes without launching a display window (headless). |

### AC-REP — Reproducibility

| ID | Criterion |
|---|---|
| **AC-REP-001** | Two sequential runs with identical config produce `best.pt` files whose `model_state` dicts are numerically identical (all tensor values equal). |

---

## 20. Project Folder Structure

```
mnist-sdd/
│
├── config/
│   └── default.yaml                  # master config (§14)
│
├── src/
│   ├── config.py                     # load_config()
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataloader.py             # get_dataloaders()
│   │   └── transforms.py            # transform pipeline builders
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── cnn.py                    # MNISTConvNet
│   │   └── mlp.py                    # MNISTNet
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Trainer
│   │   ├── early_stopping.py         # EarlyStopping
│   │   └── metrics.py               # accuracy, confusion_matrix, macro_f1
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py              # Predictor, Prediction
│   │
│   ├── tuning/
│   │   ├── __init__.py
│   │   └── tuner.py                  # HyperparamTuner
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── plot_curves.py            # loss + accuracy curves
│       ├── plot_confusion.py         # confusion matrix heatmap
│       └── plot_samples.py          # prediction grid + misclassified
│
├── scripts/
│   ├── train.py                      # CLI: train + evaluate + visualize
│   ├── evaluate.py                   # CLI: evaluate a checkpoint
│   ├── infer.py                      # CLI: inference
│   ├── tune.py                       # CLI: hyperparameter search
│   └── visualize.py                  # CLI: generate plots for a run
│
├── tests/
│   ├── conftest.py                   # shared fixtures (synthetic data, tmp dirs)
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_cnn.py
│   ├── test_mlp.py
│   ├── test_trainer.py
│   ├── test_metrics.py
│   ├── test_predictor.py
│   └── test_visualization.py
│
├── data/                             # MNIST download target  [git-ignored]
├── checkpoints/                      # model checkpoints       [git-ignored]
├── runs/                             # logs, CSVs, plots       [git-ignored]
│   └── {run_id}/
│       ├── config.yaml
│       ├── metrics.csv
│       ├── run.log
│       ├── test_results.json
│       └── plots/
│           ├── loss_curves.png
│           ├── accuracy_curve.png
│           ├── confusion_matrix.png
│           ├── sample_predictions.png
│           ├── misclassified.png
│           └── per_class_acc.png
│
├── .claude/                          # Claude Code tooling     [git-ignored]
│   └── commands/
│       └── git-commit.md
│
├── CLAUDE.md                         # developer guide
├── SPEC.md                           # this document
├── DEVLOG.md                         # session log
├── requirements.txt                  # pinned runtime deps
├── requirements-dev.txt              # test + dev deps
└── .gitignore
```

---

## 21. Out of Scope

The following items are explicitly excluded from this version of the system. A future revision of this document is required to bring any item into scope.

| Item | Rationale |
|---|---|
| Multi-GPU / distributed training | Out of scope for single-workstation research use |
| ONNX / TorchScript export | No production deployment requirement |
| REST API or web frontend | No serving requirement |
| Custom dataset (non-MNIST) | Architecture is MNIST-specific in its default config |
| Federated learning | No multi-party requirement |
| Mixed-precision (FP16) training | Unnecessary for this dataset scale |
| TensorBoard / W&B integration | CSV + matplotlib sufficient for v1 |

---

## 22. Change Log

| Version | Date | Author | Summary |
|---|---|---|---|
| 1.0.0 | 2026-05-18 | Ghanshyam | Initial draft: MLP-only dense network, 8 sections |
| 2.0.0 | 2026-05-18 | Ghanshyam | Full enterprise rewrite: CNN added as primary architecture, MLP retained as baseline; 22 sections; formal FR/NFR/AC numbering; full config schema; reproducibility, dependency, and visualization chapters added |

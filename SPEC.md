# SPEC.md — MNIST Dense Neural Network

> Status tags: `[ ]` todo · `[~]` in-progress · `[x]` done
>
> All behaviour described here is the authoritative contract.
> Implementation must match this document; any deviation requires a spec update first.

---

## 0. Config File (`config/default.yaml`) `[ ]`

The config file is the single control plane for every pipeline.
No pipeline may hard-code a value that appears in this file.

### 0.1 Schema skeleton

```yaml
# ── Reproducibility ──────────────────────────────────────────────────────────
seed: 42

# ── Data ─────────────────────────────────────────────────────────────────────
data:
  root: ./data            # where torchvision downloads MNIST
  batch_size: 64
  num_workers: 2
  val_split: 0.1          # fraction of training set used for validation
  normalize:
    mean: 0.1307
    std:  0.3081

# ── Model ─────────────────────────────────────────────────────────────────────
model:
  input_size: 784         # 28×28 flattened
  hidden_layers:          # list of layer widths
    - 256
    - 128
  num_classes: 10
  activation: relu        # relu | tanh | sigmoid
  dropout: 0.2

# ── Training ──────────────────────────────────────────────────────────────────
training:
  epochs: 20
  optimizer: adam         # adam | sgd
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  lr_scheduler:
    type: step            # step | cosine | none
    step_size: 5
    gamma: 0.5
  early_stopping:
    enabled: true
    patience: 5
    monitor: val_loss     # val_loss | val_acc
  checkpoint:
    dir: ./checkpoints
    save_best_only: true
  log_dir: ./runs

# ── Inference ─────────────────────────────────────────────────────────────────
inference:
  checkpoint: ./checkpoints/best.pt
  device: cpu             # cpu | cuda | auto

# ── Hyperparameter Tuning ─────────────────────────────────────────────────────
tuning:
  n_trials: 50
  direction: maximize     # maximize val_acc
  pruner: median          # median | none
  storage: null           # optuna DB URI or null for in-memory
  search_space:
    learning_rate:
      type: float_log
      low: 1.0e-4
      high: 1.0e-2
    batch_size:
      type: categorical
      choices: [32, 64, 128]
    hidden_layers:
      type: categorical
      choices:
        - [128]
        - [256, 128]
        - [512, 256, 128]
    dropout:
      type: float
      low: 0.0
      high: 0.5
    activation:
      type: categorical
      choices: [relu, tanh]
```

### 0.2 Config loading contract

- `src/config.py` exposes a single function `load_config(path: str) -> dict`.
- The returned dict is plain Python (not a namespace object).
- All CLI entry points accept `--config <path>` (default: `config/default.yaml`).

---

## 1. Data Loading Pipeline `[ ]`

**Goal**: produce ready-to-use `DataLoader` objects for train, validation, and test splits.

### 1.1 Data source

- Dataset: MNIST via `torchvision.datasets.MNIST`.
- Automatically downloaded to `data.root` on first run if not present.

### 1.2 Splits

| Split      | Source              | Size          |
|------------|---------------------|---------------|
| train      | MNIST train (60 k)  | `(1 − val_split) × 60 k` |
| validation | MNIST train (60 k)  | `val_split × 60 k` |
| test       | MNIST test (10 k)   | 10 000 (fixed) |

- Train/validation split: random, stratified by class label.
- Split is reproducible given the same `seed`.

### 1.3 Transforms

- Convert PIL image → `torch.FloatTensor`.
- Normalize with `(mean, std)` from config.
- No data augmentation in the baseline; augmentation may be added as a config flag later.

### 1.4 Public API

```python
# src/data/dataloader.py
def get_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
```

### 1.5 Acceptance criteria

- [ ] Each loader yields `(images, labels)` where `images.shape == (B, 784)` and `labels.shape == (B,)`.
- [ ] Class distribution in train and val splits matches the full training set (stratified).
- [ ] Running `get_dataloaders` twice with the same seed yields identical splits.

---

## 2. Model Architecture `[ ]`

**Goal**: a configurable fully-connected DNN for 10-class classification.

### 2.1 Architecture

- Input layer: 784 units (flattened 28×28 grayscale image).
- Hidden layers: variable depth and width, read from `model.hidden_layers`.
- Output layer: 10 units (raw logits, no softmax — use `CrossEntropyLoss`).
- Between each pair of layers: `BatchNorm1d` → activation → `Dropout`.

### 2.2 Public API

```python
# src/model/network.py
class MNISTNet(nn.Module):
    def __init__(self, cfg: dict) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 784) → logits: (B, 10)"""
```

### 2.3 Acceptance criteria

- [ ] `MNISTNet(cfg).forward(torch.randn(4, 784)).shape == (4, 10)`.
- [ ] Number of hidden layers matches `len(cfg['model']['hidden_layers'])`.
- [ ] Dropout is disabled at inference time (`model.eval()`).

---

## 3. Training & Validation Pipeline `[ ]`

**Goal**: train the model, validate each epoch, log metrics, apply early stopping, and save checkpoints.

### 3.1 Training loop

- One epoch = one full pass over the training `DataLoader`.
- Loss function: `nn.CrossEntropyLoss`.
- Optimizer: configured by `training.optimizer` (adam or sgd) with `learning_rate` and `weight_decay`.
- LR scheduler: stepped after each epoch according to `training.lr_scheduler`.

### 3.2 Validation loop

- Run after every training epoch.
- Compute: `val_loss` (mean cross-entropy) and `val_acc` (% correct).
- No gradient computation (`torch.no_grad`).

### 3.3 Metrics logged per epoch

| Metric       | Description                     |
|--------------|---------------------------------|
| `train_loss` | Mean cross-entropy over train set |
| `val_loss`   | Mean cross-entropy over val set   |
| `val_acc`    | Accuracy on val set (0–1)         |
| `lr`         | Current learning rate             |

Metrics are printed to stdout and written to `{log_dir}/{run_id}/metrics.csv`.

### 3.4 Checkpointing

- Save model state dict to `{checkpoint_dir}/{run_id}/epoch_{n}.pt` each epoch (or best-only if configured).
- Always save `best.pt` pointing to the checkpoint with the best `monitor` metric.
- Checkpoint file contains: `{'epoch': int, 'model_state': ..., 'optimizer_state': ..., 'metrics': dict}`.

### 3.5 Early stopping

- Stop training if the monitored metric does not improve for `patience` consecutive epochs.
- "Improve" means: decrease for `val_loss`, increase for `val_acc`.

### 3.6 Public API

```python
# src/training/trainer.py
class Trainer:
    def __init__(self, model: nn.Module, cfg: dict, run_id: str) -> None: ...
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Train for configured epochs. Return dict of final metrics."""
```

### 3.7 Acceptance criteria

- [ ] `metrics.csv` exists after training with correct column headers.
- [ ] `best.pt` is saved and loadable.
- [ ] Training stops early when patience is exhausted.
- [ ] Resuming from a checkpoint reproduces the same final metrics (given same seed).

---

## 4. Inference Pipeline `[ ]`

**Goal**: load a trained checkpoint and classify new images reliably.

### 4.1 Input formats

| Mode       | Input                                  |
|------------|----------------------------------------|
| Single     | Path to a PNG/JPG grayscale image      |
| Batch      | Path to a directory of images          |
| Tensor     | `torch.Tensor` of shape `(B, 1, 28, 28)` or `(B, 784)` |

### 4.2 Preprocessing

- Resize to 28×28 if needed.
- Convert to grayscale if input is RGB.
- Apply the same normalization as training (parameters come from config).

### 4.3 Output

```python
@dataclass
class Prediction:
    label: int            # predicted digit 0–9
    confidence: float     # max softmax probability
    probabilities: list[float]  # softmax over all 10 classes
```

### 4.4 Public API

```python
# src/inference/predictor.py
class Predictor:
    def __init__(self, cfg: dict) -> None: ...  # loads checkpoint from cfg

    def predict_image(self, image_path: str) -> Prediction: ...
    def predict_batch(self, image_dir: str) -> list[Prediction]: ...
    def predict_tensor(self, x: torch.Tensor) -> list[Prediction]: ...
```

### 4.5 Acceptance criteria

- [ ] `predict_tensor(torch.zeros(1, 784))` returns a `Prediction` without error.
- [ ] Predicted label matches `torch.argmax(probabilities)`.
- [ ] Model is in `eval()` mode; no gradient computation.

---

## 5. Hyperparameter Tuning Pipeline `[ ]`

**Goal**: use Optuna to search for the best hyperparameters and report the optimal config.

### 5.1 Search space

Defined under `tuning.search_space` in the config (see §0.1).
Supported parameter types:

| Type          | Optuna call                      |
|---------------|----------------------------------|
| `float`       | `trial.suggest_float`            |
| `float_log`   | `trial.suggest_float(..., log=True)` |
| `int`         | `trial.suggest_int`              |
| `categorical` | `trial.suggest_categorical`      |

### 5.2 Objective function

- For each trial: build a model and trainer from the trial's sampled config, train for `training.epochs` epochs (or until early stopping), return the best `val_acc`.
- Each trial uses the same train/val split (fixed seed) so trials are comparable.

### 5.3 Study configuration

- `n_trials`, `direction`, `pruner`, and `storage` come from config.
- Median pruner halts unpromising trials after each epoch (Optuna `MedianPruner`).

### 5.4 Output

- Best hyperparameters printed to stdout.
- Full study saved to `{log_dir}/tuning/study.pkl`.
- Best config written to `{log_dir}/tuning/best_config.yaml`.

### 5.5 Public API

```python
# src/tuning/tuner.py
class HyperparamTuner:
    def __init__(self, base_cfg: dict) -> None: ...
    def run(self) -> dict:
        """Run Optuna study. Return best hyperparameters as a dict."""
```

### 5.6 Acceptance criteria

- [ ] A 2-trial run completes without error.
- [ ] `best_config.yaml` is valid YAML and loadable as a config.
- [ ] Trials with early stopping still report a valid `val_acc` objective.

---

## 6. CLI Entry Points `[ ]`

| Script              | Flags                                              |
|---------------------|----------------------------------------------------|
| `scripts/train.py`  | `--config`, `--run-id` (auto-generated if omitted) |
| `scripts/evaluate.py` | `--config`, `--checkpoint`                       |
| `scripts/infer.py`  | `--config`, `--checkpoint`, `--image` / `--image-dir` |
| `scripts/tune.py`   | `--config`                                         |

All scripts exit with code 0 on success, non-zero on error.

---

## 7. Testing `[ ]`

- Unit tests live in `tests/`; each source module has a corresponding test file.
- Tests must not download real MNIST data; use synthetic tensors or a tiny fixture dataset.
- Run with: `pytest tests/ -v`.

| Test file          | Covers                                         |
|--------------------|------------------------------------------------|
| `test_data.py`     | Loader shapes, split sizes, reproducibility    |
| `test_model.py`    | Forward pass shape, layer count, eval dropout  |
| `test_trainer.py`  | One-epoch train loop, checkpoint save/load     |
| `test_predictor.py`| All three predict modes, output types          |

---

## 8. Out of Scope (for this version)

- Convolutional layers (CNN) — this spec targets dense networks only.
- Data augmentation (random crops, flips).
- Multi-GPU or distributed training.
- ONNX / TorchScript export.
- A web or GUI frontend.

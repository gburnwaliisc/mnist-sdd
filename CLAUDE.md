# CLAUDE.md — MNIST SDD Project Development Memory

> This file is the living institutional memory of the project.
> It records *why* decisions were made, not just *what* was decided.
> Update it whenever a non-obvious decision is taken, a design is changed,
> or a debugging session reveals something worth remembering.
> The SPEC.md is the contract. This file is the reasoning behind it.

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Session Continuity Log](#2-session-continuity-log)
3. [Architectural Decisions](#3-architectural-decisions)
4. [Technology Choices and Rationale](#4-technology-choices-and-rationale)
5. [CNN Design Decisions](#5-cnn-design-decisions)
6. [Optimizer and Scheduler Decisions](#6-optimizer-and-scheduler-decisions)
7. [Data Pipeline Decisions](#7-data-pipeline-decisions)
8. [Assumptions](#8-assumptions)
9. [Coding Conventions and Rationale](#9-coding-conventions-and-rationale)
10. [Debugging Considerations and Known Gotchas](#10-debugging-considerations-and-known-gotchas)
11. [Dependency Notes](#11-dependency-notes)
12. [Future Improvements](#12-future-improvements)
13. [Key Commands](#13-key-commands)
14. [Spec-Driven Process Rules](#14-spec-driven-process-rules)

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project name** | MNIST Handwritten Digit Classification System |
| **Spec document** | `SPEC.md` (SDD-MNIST-001 v2.0.0) |
| **Repository** | https://github.com/gburnwaliisc/mnist-sdd |
| **Language** | Python 3.10+ |
| **Primary framework** | PyTorch 2.1+ |
| **Development method** | Specification-Driven Development (SDD) |
| **Author** | Ghanshyam, Indian Institute of Science |
| **Started** | 2026-05-18 |

### What this project is

A research-grade, fully reproducible pipeline for MNIST digit classification using a Convolutional Neural Network (CNN) as the primary model and a Multi-Layer Perceptron (MLP) as a comparative baseline. Every parameter is config-driven; every decision is documented here.

### What this project is not

- A production serving system (no REST API, no containerization in v1).
- A benchmark against state-of-the-art (≥99% on MNIST is the target, not squeezing out 99.7%).
- A generalizable image classification framework — architecture choices are MNIST-specific.

---

## 2. Session Continuity Log

### Session 1 — 2026-05-18

**What was done:**
- Initialized project with SDD methodology.
- Created CLAUDE.md v1 (developer guide, now superseded by this document).
- Created SPEC.md v1.0.0 covering dense-network-only pipelines.
- Initialized git repository; set local identity to `ghanshyamb@iisc.ac.in`.
- Pushed to GitHub: `gburnwaliisc/mnist-sdd`.
- Created project-scoped `/git-commit` skill at `.claude/commands/git-commit.md`.
- Created global `/git-commit` skill at `~/.claude/commands/git-commit.md`.
- Added `.gitignore` covering `.claude/`, `data/`, `checkpoints/`, `runs/`.
- Created `DEVLOG.md` recording session-1 conversation.

**Key decision made this session:**
- Upgraded from dense-network-only to CNN-primary + MLP-baseline after reviewing literature and project plan. SPEC.md v2.0.0 reflects this change.

**State at end of session:**
- SPEC.md v2.0.0 complete (enterprise format, 22 sections, 60+ requirements, 30+ acceptance criteria).
- CLAUDE.md v2 (this file) complete.
- **No implementation code written.** All files are specification and tooling only.

**Where to pick up next:**
- Phase 0: Implement `config/default.yaml` and `src/config.py` (`load_config`).
- Verify `AC-CFG-001` and `AC-CFG-002` pass before moving to Phase 1.

---

## 3. Architectural Decisions

### 3.1 CNN as Primary, MLP as Baseline — Not CNN-only

**Decision:** Implement both `MNISTConvNet` (primary) and `MNISTNet` (MLP baseline). Selection via `model.arch` in config.

**Reasoning:**
- The original spec targeted a dense network. Switching to CNN-only would discard the initial educational objective of comparing architectures.
- A shared `Trainer` and `Predictor` that works for both architectures costs nothing in complexity but gives two data points: how much spatial structure matters for MNIST.
- The MLP baseline at ≥97.5% accuracy is a meaningful sanity check. If the CNN underperforms the MLP, something is wrong.

**What was rejected:** A single unified model class with a flag. Keeping them in separate files (`cnn.py`, `mlp.py`) makes each architecture readable in isolation and prevents the kind of tangled conditional logic that becomes unmaintainable.

---

### 3.2 Single YAML Config as Exclusive Control Plane

**Decision:** One `config/default.yaml`. No module may contain hard-coded values that appear in it.

**Reasoning:**
- Reproducibility is the core non-functional requirement. A run is only reproducible if every parameter that could affect it is captured. YAML is human-readable, diffable, and trivially serializable — it can be saved alongside checkpoints to make runs self-describing.
- The alternative (argparse-only CLI flags) would mean the config is implicit in the shell history, which cannot be committed or attached to a checkpoint.

**Consequence:** The config snapshot saved to `runs/{run_id}/config.yaml` at training start is the ground truth for what any given checkpoint was trained with. When debugging a checkpoint from two weeks ago, you look at that file, not at your shell history.

---

### 3.3 Config Sub-dicts Passed Explicitly, Not Imported Globally

**Decision:** Each module receives only its own config sub-dict as an argument. `training/trainer.py` receives `cfg['training']`, not the full config dict.

**Reasoning:**
- Prevents hidden coupling between pipeline modules. If the trainer reaches into `cfg['data']` directly, changing the data config schema silently breaks the trainer — exactly the kind of bug that takes hours to locate.
- Makes module interfaces explicit and testable in isolation: you can instantiate a `Trainer` in a test by passing a minimal dict without constructing the full config.

**Exception:** `src/config.py`'s `load_config()` and the CLI entry points in `scripts/` see the full config — they are the integration layer and that is their job.

---

### 3.4 Logits Out, No Softmax in the Model

**Decision:** Both `MNISTConvNet` and `MNISTNet` output raw logits. Softmax is never applied inside the model.

**Reasoning:**
- `nn.CrossEntropyLoss` expects logits and applies `log_softmax` internally. Applying softmax before the loss would compute `log(softmax(x))` which is numerically inferior to `log_softmax(x)` (loses precision for small values).
- During inference, `torch.nn.functional.softmax` is applied in the `Predictor` to produce probabilities — this is the correct, single place to do it.
- This is a well-known PyTorch pattern. Violating it causes subtle numerical errors that are hard to debug because the model still trains, just slightly worse.

---

### 3.5 `best.pt` + `last.pt` Always Saved

**Decision:** Always write both `best.pt` (best monitored metric) and `last.pt` (end of every epoch), regardless of `save_best_only`.

**Reasoning:**
- `best.pt` is for evaluation and inference — you want the model that performed best on validation.
- `last.pt` is for resuming — if training is interrupted (machine shutdown, OOM), you resume from `last.pt`, not `best.pt`. If you only have `best.pt` and training was interrupted at epoch 15 (best was epoch 8), you cannot resume without re-running the first 8 epochs.
- The `save_every_epoch` flag is optional and off by default. It creates one file per epoch which can consume significant disk space (each `.pt` is typically 1–20 MB depending on architecture).

---

## 4. Technology Choices and Rationale

### 4.1 PyTorch over TensorFlow / JAX

**Choice:** PyTorch 2.1+

**Reasoning:**
- Dynamic computation graph (eager mode) makes debugging straightforward: you can insert `print` statements or a `pdb` breakpoint anywhere in `forward()` and inspect actual tensor values.
- `torch.compile()` (available since PyTorch 2.0) provides a path to compiled performance without changing the programming model — relevant if this project is extended to larger datasets.
- PyTorch is the de facto standard in academic research (IISc context), meaning published baselines, pretrained weights, and tooling (Optuna's PyTorch integration, torchvision) are first-class.
- TensorFlow 2.x is a reasonable alternative but the IISc ML ecosystem is primarily PyTorch.
- JAX would offer better XLA compilation but the ecosystem for MNIST-class projects (dataset loading, training loops) is less mature and would add friction for no benefit at this scale.

---

### 4.2 Optuna over Ray Tune / Hyperopt / Manual Grid Search

**Choice:** Optuna 3.0+

**Reasoning:**
- Optuna uses Tree-structured Parzen Estimator (TPE) by default — a Bayesian optimization method that outperforms grid search and random search for moderate numbers of trials (20–100).
- The `MedianPruner` terminates unpromising trials early using the median of intermediate values reported via `trial.report()`. For a 20-epoch training run, this can reduce tuning wall-clock time by 40–60%.
- The search space is defined in the config YAML (see `tuning.search_space`) rather than in Python code. This means the search space is versioned with the experiment, not buried in a script.
- Optuna's in-memory storage (`storage: null`) requires no external database, making it zero-dependency for a laptop/workstation setup.
- Ray Tune is more powerful for distributed tuning but adds significant infrastructure overhead not justified for single-machine MNIST experiments.

---

### 4.3 PyYAML over TOML / JSON / Hydra / OmegaConf

**Choice:** PyYAML with a thin `load_config()` wrapper

**Reasoning:**
- YAML supports comments, which is essential for a config file that serves as documentation (every key in `default.yaml` should have an inline comment explaining its effect).
- JSON supports no comments and is verbose for nested configs.
- TOML is readable but PyYAML has broader ecosystem support and is already a torchvision dependency.
- Hydra/OmegaConf are powerful but introduce substantial framework-level complexity (composition, overrides, multirun) that is overkill for a single-project config. The SDD principle is to add complexity only when it earns its place.
- `load_config()` returns a plain Python `dict`. This was a deliberate choice over `OmegaConf.DictConfig` or a namespace object — plain dicts are transparent, JSON-serializable, and do not surprise readers with proxy object behaviour.

---

### 4.4 scikit-learn for Stratified Split

**Choice:** `sklearn.model_selection.StratifiedShuffleSplit`

**Reasoning:**
- PyTorch's `random_split` does not support stratification. A naive random split on 60,000 samples would be approximately stratified by the law of large numbers, but the spec requires it to be guaranteed.
- `StratifiedShuffleSplit` with `n_splits=1` is the standard way to produce a single reproducible stratified split from a labeled dataset. The `random_state` parameter maps directly to `config.seed`.
- Introducing scikit-learn adds a dependency, but it is already in the scientific Python stack that any PyTorch user will have installed. It is not a heavyweight addition.

---

### 4.5 matplotlib for Visualization

**Choice:** matplotlib with `Agg` backend

**Reasoning:**
- matplotlib is the universal scientific plotting library in Python; any reader of the plots knows how they were made.
- The `Agg` backend renders to PNG without requiring a display server. This is essential for running on headless machines (remote servers, CI pipelines).
- Seaborn was considered for the confusion matrix heatmap but rejected to minimize dependencies. matplotlib's `imshow` with appropriate annotations achieves the same result.

---

### 4.6 pandas for CSV Metrics

**Choice:** `pandas.DataFrame.to_csv()`

**Reasoning:**
- Writing CSV row-by-row with Python's `csv` module is error-prone (easy to forget to flush, easy to get column order wrong on append).
- pandas provides a clean `DataFrame.to_csv(mode='a', header=False)` for append-mode writes, and `to_csv(mode='w', header=True)` for the first epoch.
- pandas is already a transitive dependency of Optuna and scikit-learn, so it adds no new installation overhead.

---

## 5. CNN Design Decisions

### 5.1 Two Conv Blocks, Not Three or One

**Decision:** Default `conv_channels: [32, 64]` — two convolutional blocks.

**Reasoning:**
- One conv block cannot learn hierarchical features. MNIST digits require recognizing strokes (low-level) and their spatial arrangement (higher-level). Two blocks correspond to the LeNet-5 design that was purpose-built for MNIST.
- Three or more conv blocks on 28×28 images hit the spatial dimension floor quickly. After two `MaxPool2d(2,2)` operations the feature map is 7×7. A third pooling would bring it to 3×3, which is aggressive for a 28×28 input and can hurt gradient flow.
- The configurable `conv_channels` list means the number of blocks can be increased by the user; the default of two is the validated baseline.

---

### 5.2 3×3 Kernels with `padding=1`

**Decision:** All conv layers use `kernel_size=3, padding=1`.

**Reasoning:**
- 3×3 kernels are the de facto standard since VGG (2014). Two stacked 3×3 convolutions have the same receptive field as one 5×5 convolution but fewer parameters and an additional non-linearity.
- `padding=1` preserves spatial dimensions through the conv operation (`H_out = H_in`), keeping spatial resolution management explicit and predictable — only `MaxPool2d` reduces dimensions.
- 5×5 kernels (used in original LeNet-5) would work but use more parameters for no measurable benefit on MNIST at this scale.

---

### 5.3 MaxPool2d not AvgPool2d or Strided Convolution

**Decision:** Downsampling via `MaxPool2d(2, 2)`.

**Reasoning:**
- Max pooling retains the strongest activation in each local region, which is appropriate for digit features (the presence of a stroke is more informative than its average intensity).
- Strided convolutions (used in ResNets) are learnable downsampling but add parameters and complexity. For MNIST at this scale, fixed max pooling is sufficient.
- Average pooling would suppress strong edge detections, which are critical features for digit boundaries.

---

### 5.4 BatchNorm Before Activation, After Conv

**Decision:** Conv → BatchNorm → Activation (the original BN paper order).

**Reasoning:**
- BatchNorm normalizes the pre-activation distribution, which stabilizes training by preventing the activation inputs from drifting into saturation regions.
- The alternative (Conv → Activation → BatchNorm, as used in some modern architectures) normalizes post-activation values. For ReLU this can cause the mean to be consistently positive, partially defeating the purpose of zero-centering. For MNIST at this scale the difference is negligible, but the original order is the safer default.
- BatchNorm is made optional via `model.use_batchnorm` so ablations can be run easily.

---

### 5.5 Dropout Only in the FC Layer, Not After Conv Blocks

**Decision:** Dropout is applied only after the FC hidden layer. Conv block activations are not dropped.

**Reasoning:**
- Spatial dropout (dropping entire feature map channels) is used in larger convolutional networks. For a two-block CNN on 28×28 input, the feature maps are small enough (32×14×14 and 64×7×7) that standard dropout on the flattened FC layer provides sufficient regularization.
- Applying per-element dropout to conv feature maps with MNIST's smooth handwriting patterns can destroy spatially correlated features (a stroke passing through the dropout mask appears broken to the next layer). The benefit does not justify this risk.
- The default `dropout: 0.5` on the FC layer is high by modern standards but appropriate for the FC bottleneck that compresses 3,136 features to 10 classes.

---

### 5.6 ReLU as Default Activation, GELU as Alternative

**Decision:** Default `activation: relu`. GELU supported as an alternative.

**Reasoning:**
- ReLU is computationally trivial (a threshold operation), interpretable, and trains well on MNIST. There is no reason to use a more complex activation as the default.
- GELU (Gaussian Error Linear Unit, used in Transformers and BERT) is smoother and has been shown to outperform ReLU in certain regimes. It is included as an option for the hyperparameter tuner to explore.
- Sigmoid and tanh are retained for the MLP (as `tanh`) for historical comparison but are not offered for the CNN — their vanishing gradient problem is more acute in deeper networks.

---

## 6. Optimizer and Scheduler Decisions

### 6.1 Adam as Default Optimizer

**Decision:** `optimizer: adam` with `lr=1e-3, weight_decay=1e-4`.

**Reasoning:**
- Adam adapts the learning rate per parameter using first and second moment estimates. For a network being trained from scratch on a well-conditioned dataset like MNIST, Adam converges reliably in under 20 epochs without extensive LR tuning.
- The default `lr=1e-3` is the Kingma & Ba (2015) recommended default and works well for this architecture and batch size.
- `weight_decay=1e-4` applies L2 regularization via Adam's weight decay term. This is a mild regularizer appropriate for a model that is not severely overparameterized relative to the MNIST training set.

**When to prefer SGD:**
- SGD with momentum and a well-tuned LR schedule can generalize slightly better than Adam on image classification (noted empirically since 2017). If the final accuracy target of ≥99% proves difficult with Adam, SGD + cosine annealing is the first thing to try.
- SGD is included as a config option precisely for this reason.

---

### 6.2 Cosine Annealing as Default Scheduler

**Decision:** `lr_scheduler.type: cosine` as default.

**Reasoning:**
- Cosine annealing decays the LR smoothly from `lr_max` to 0 over `T_max` epochs without requiring manual step-size tuning. A StepLR scheduler requires choosing both `step_size` and `gamma` — two more hyperparameters that interact with the choice of optimizer and LR.
- For Adam, cosine annealing is particularly effective in the final epochs: the decreasing LR compensates for Adam's tendency to take increasingly large steps as gradient estimates stabilize.
- `ReduceLROnPlateau` is the most principled choice (reduces LR when val_loss stops improving) but interacts awkwardly with early stopping — both react to the same signal and can interfere. Cosine annealing is schedule-based and does not interact with early stopping logic.

---

### 6.3 Early Stopping Monitors `val_loss`, Not `val_acc`

**Decision:** Default `early_stopping.monitor: val_loss`.

**Reasoning:**
- `val_loss` is a continuous signal; `val_acc` is quantized to multiples of `1/n_val_samples`. A model can improve its loss for several epochs while its accuracy stays flat (because rounded probabilities haven't crossed the decision boundary yet). Monitoring loss avoids spurious stops.
- For MNIST at high accuracy (≥98%), the accuracy signal is almost always at plateau — the difference between 98.83% and 98.91% is not visible in the accuracy signal but is captured in the loss.
- `val_acc` monitoring is offered as a config option for cases where the loss landscape is unusual (e.g., very imbalanced classes in other tasks).

---

### 6.4 Early Stopping Patience of 5 Epochs

**Decision:** `early_stopping.patience: 5`

**Reasoning:**
- With cosine annealing over 20 epochs, the LR has a natural descent shape. A patience of 5 means we tolerate a 25% window of non-improvement before halting. This is conservative enough to survive a local plateau mid-training.
- A patience of 2–3 would trigger too early during the cosine LR descent, where small LR reductions can temporarily stall improvement.
- A patience of 8–10 would negate the benefit of early stopping for a 20-epoch budget.

---

## 7. Data Pipeline Decisions

### 7.1 Stratified Split Using StratifiedShuffleSplit

**Decision:** Enforce stratification on the train/validation split.

**Reasoning:**
- MNIST classes are approximately balanced (~6,000 per class in 60,000 samples), so naive random splitting would be close to stratified. However, "close to" is not the same as guaranteed, and the SPEC requires the split to be provably stratified.
- Stratification matters more in practice when `val_split` is small (e.g., 0.05 → 3,000 samples). Without stratification, you could randomly get 200 examples of one class and 450 of another in the validation set, making per-class validation accuracy misleading.

---

### 7.2 Normalization Constants Are Fixed, Not Computed at Runtime

**Decision:** `mean=0.1307, std=0.3081` are treated as constants from config, not recomputed.

**Reasoning:**
- These are the well-established, widely-cited channel statistics for the MNIST training set. Recomputing them at runtime adds a pass over 60,000 images for no benefit — the values are stable across all seeds and have been verified by the community for decades.
- If a non-standard MNIST subset is used (e.g., a custom val/train resplit), recomputing would give marginally different values that would not match values used by published benchmarks, making comparison harder.
- If this project were extended to a different dataset, the config would be updated with that dataset's statistics — the mechanism is already in place.

---

### 7.3 Augmentation Off by Default

**Decision:** `data.augmentation: false` in default config. Random affine augmentation supported but disabled.

**Reasoning:**
- MNIST at 60,000 training samples is not data-scarce. The CNN achieves ≥99% without augmentation. Adding augmentation increases training time and introduces two more hyperparameters (degree range, translate range) without a measurable benefit at the target accuracy level.
- Augmentation is included as a config flag (not hardcoded off) so that experiments with it can be run and compared cleanly.
- If this project were extended to Fashion-MNIST or similar datasets where augmentation helps, flipping the flag is the only change required.

---

## 8. Assumptions

These assumptions were made during specification. If any prove false, the SPEC and this document must be updated.

| # | Assumption | Risk if false |
|---|---|---|
| A-01 | The execution environment has internet access for the first MNIST download. | The `torchvision` download will fail silently or with an unhelpful error. Mitigation: pre-download and set `data.root` to the local path. |
| A-02 | Python 3.10+ is available. | `tuple[...]` type hints and `match` statements will cause `SyntaxError` on Python 3.9 and below. |
| A-03 | The machine has at least 2 GB of free RAM. | Holding 60,000 images in memory as float32 tensors requires ~188 MB. With model, optimizer, and batch buffers, 2 GB is comfortable. 512 MB might be tight. |
| A-04 | MNIST data at `data.root` is not modified between runs. | Stratified splits are reproducible given the same seed and the same data ordering. If the dataset is re-downloaded or reordered, seeds produce different splits. |
| A-05 | `num_workers > 0` requires a POSIX-compatible multiprocessing backend. | On Windows with `num_workers > 0`, PyTorch DataLoader uses `spawn` start method, which requires scripts to be guarded with `if __name__ == '__main__':`. CLI scripts shall include this guard. |
| A-06 | The CNN default configuration (`conv_channels=[32,64], fc_hidden=256`) fits in memory on a laptop CPU without OOM. | A forward pass for batch size 64 through the default CNN requires ~50 MB of activation memory. This is well within any modern laptop's capacity. |
| A-07 | Optuna in-memory storage (`storage: null`) is acceptable for single-machine tuning runs. | If the process is killed mid-study, all trial data is lost. For long tuning runs (>50 trials), setting `storage` to an SQLite URI is strongly recommended. |
| A-08 | The test set is treated as a true holdout — it is never inspected during development. | Any decision made by examining test set performance (e.g., tuning `dropout` because test acc was low) constitutes data leakage and invalidates the reported metric. |

---

## 9. Coding Conventions and Rationale

### 9.1 Type Annotations on All Public Interfaces

All public functions and class methods shall have full type annotations. Private helpers (prefixed `_`) should have annotations but are not required to.

**Why:** Type annotations serve as inline documentation and enable static analysis with `mypy`. For a training pipeline where tensor shapes are critical, annotating `x: torch.Tensor` and return types is the minimum — annotating with `# (B, 1, 28, 28)` in docstrings is preferred where shapes are non-obvious.

### 9.2 `pathlib.Path` for All File Paths

Use `pathlib.Path` everywhere. Never use string concatenation to build paths.

**Why:** Windows uses backslashes; Linux/macOS use forward slashes. `pathlib.Path` handles this transparently. Since the SPEC requires Windows/Linux/macOS portability (NFR-PRT-001), string paths are a latent cross-platform bug.

### 9.3 No Global State Outside Config Dict

No module-level mutable variables. All state flows through the config dict and explicit arguments.

**Why:** Global state makes modules non-reentrant and breaks the hyperparameter tuner — Optuna runs multiple trials in the same process; any module-level state set in trial N leaks into trial N+1.

### 9.4 `torch.no_grad()` Context Manager in All Eval Paths

Every validation loop, test loop, and inference call shall be wrapped in `torch.no_grad()`.

**Why:** Without `no_grad()`, PyTorch allocates gradient buffers for every forward operation. Over a full validation pass (6,000 samples), this can consume several hundred MB of additional memory that is never freed until the context is exited. On memory-constrained machines this causes OOM on the validation pass, not the training pass — which is a confusing failure mode.

### 9.5 `model.eval()` / `model.train()` Discipline

`model.eval()` shall be called before any validation, test, or inference pass. `model.train()` shall be called at the start of each training epoch.

**Why:** `model.eval()` switches BatchNorm from using batch statistics to running statistics, and sets dropout probability to zero. Forgetting to call `model.train()` after a validation pass means the entire next training epoch runs with dropout disabled — the model trains without regularization but you don't know it. Forgetting `model.eval()` before inference means predictions have random dropout applied, making confidence scores non-deterministic.

### 9.6 One `__init__.py` per `src/` Subdirectory

All subdirectories under `src/` shall be proper Python packages with `__init__.py` files.

**Why:** Enables clean imports (`from src.model.cnn import MNISTConvNet`) from the project root and from test files, without requiring `sys.path` manipulation.

---

## 10. Debugging Considerations and Known Gotchas

### 10.1 MNIST Download Failure

**Symptom:** `RuntimeError: Dataset not found` or a connection timeout.
**Cause:** No internet access, or the torchvision MNIST mirror URL has changed.
**Fix:** Pre-download MNIST manually. The expected directory structure under `data.root` is:
```
data/MNIST/raw/
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
```
Set `download=False` in `torchvision.datasets.MNIST` if the files are already present.

---

### 10.2 Windows DataLoader Deadlock

**Symptom:** Training hangs indefinitely at the first batch when `num_workers > 0` on Windows.
**Cause:** Windows uses `spawn` for multiprocessing. The DataLoader worker processes re-import the main script, which re-executes the training code in the workers, causing a deadlock.
**Fix:** All CLI entry points must be guarded:
```python
if __name__ == '__main__':
    main()
```
This is already mandated by assumption A-05. If this symptom appears, look for an unguarded `main()` call.

---

### 10.3 Reproducibility Breaks Silently with `num_workers > 1`

**Symptom:** Two runs with the same seed produce different training loss curves.
**Cause:** DataLoader worker processes use the same base seed unless `worker_init_fn` is set. Multiple workers seeded identically produce correlated random numbers, and the order in which workers deliver batches is non-deterministic.
**Fix:** The `worker_init_fn` specified in `RR-007` seeds each worker with `config.seed + worker_id`, making each worker independent and the overall batch ordering deterministic given the same number of workers.

---

### 10.4 Early Stopping Triggers Immediately on First Epoch

**Symptom:** Training halts after epoch 1 with "early stopping triggered."
**Cause:** The `EarlyStopping` state is not initialized correctly. If `best_value` defaults to `None` and the comparison fails on the first epoch, or if `best_value` is initialized to `0.0` when monitoring `val_acc`, the first epoch looks like no improvement.
**Fix:** Initialize `best_value` to `float('inf')` when monitoring `val_loss` and to `float('-inf')` when monitoring `val_acc`. The first epoch always improves over these sentinels.

---

### 10.5 val_acc Plateaus While val_loss Still Decreases

**Symptom:** `val_acc` is stuck at, e.g., 98.5% for 10 epochs even though `val_loss` is decreasing.
**Cause:** This is expected behaviour, not a bug. At high accuracy levels, the classifier is correctly predicting almost all samples. Further loss reduction reflects improved confidence on already-correct predictions, not new correct predictions.
**Implication:** Do not switch to `monitor: val_acc` as a fix. This is why the default monitor is `val_loss`.

---

### 10.6 Checkpoint `model_state` Mismatches After Architecture Change

**Symptom:** `RuntimeError: Error(s) in loading state_dict` when loading a checkpoint.
**Cause:** The architecture defined in the current config does not match the architecture that produced the checkpoint (e.g., you changed `conv_channels` between training runs).
**Fix:** Every checkpoint contains a `config` dict (see §10.4.2 of SPEC). Load the checkpoint's embedded config instead of the current config:
```python
ckpt = torch.load(path)
cfg = ckpt['config']   # use this, not config/default.yaml
model = build_model(cfg)
model.load_state_dict(ckpt['model_state'])
```

---

### 10.7 Optuna `MedianPruner` Prunes All Trials

**Symptom:** Every trial is pruned after epoch 1 or 2 during hyperparameter tuning.
**Cause:** The pruner requires a reference population to compute the median. With fewer than 5 completed trials, the median is unreliable and the pruner can be overly aggressive.
**Fix:** Set `n_startup_trials` in the `MedianPruner` constructor to at least 5 (the Optuna default is 5). Verify this is respected in `tuner.py`.

---

### 10.8 `model.eval()` Forgotten Before Test Evaluation

**Symptom:** Test accuracy is lower than validation accuracy from the same checkpoint, or is non-deterministic between calls.
**Cause:** BatchNorm is using batch statistics instead of running statistics, and dropout is still active.
**Fix:** The `Trainer.evaluate()` method must call `model.eval()` as its first statement and `model.train()` as its last. Add an assertion in `test_trainer.py` that checks `model.training == False` during evaluation.

---

### 10.9 Loss NaN After First Batch

**Symptom:** `train_loss` is `nan` from the very first batch.
**Cause A:** Learning rate is too high (e.g., `lr=1.0` instead of `1e-3`). Adam with a high LR can produce parameter updates that overflow.
**Cause B:** A `log(0)` somewhere — this can happen if softmax is applied before `CrossEntropyLoss`, producing probabilities of exactly 0.0, whose log is `-inf`.
**Cause C:** Input tensor is not normalized — raw pixel values in `[0, 255]` with a mean-subtraction normalization expecting values in `[0, 1]` can cause very large activations.
**Fix:** Check the config LR value, verify no softmax is applied before the loss, and confirm the normalization transform is being applied.

---

## 11. Dependency Notes

### 11.1 PyTorch Version Lock

**Minimum: 2.1.0.** The `torch.compile()` API (used optionally for performance) stabilized in 2.0. The `weights_only=True` parameter to `torch.load()` — which protects against arbitrary code execution from malicious checkpoint files — was added in 2.0. We use it.

**Do not use `torch.load(path)` without `weights_only=True`** unless loading the full checkpoint dict that contains non-tensor objects (config dict, metrics). In that case, ensure the checkpoint source is trusted.

---

### 11.2 torchvision Version Must Match PyTorch

torchvision version compatibility is strict. Use the table at https://github.com/pytorch/vision#installation to confirm the correct pair. Mismatched versions cause silent incorrect behaviour in transforms (not necessarily an import error).

---

### 11.3 Optuna ≥ 3.0 Required

Optuna 3.0 introduced breaking changes to the `Trial.report()` and `Trial.should_prune()` APIs. The `MedianPruner` behaviour also changed. Do not use Optuna 2.x.

---

### 11.4 scikit-learn Transitive Dependency

scikit-learn is listed explicitly in `requirements.txt` even though it is a transitive dependency of other packages. Transitive dependencies can change version silently when a parent package updates. Pinning scikit-learn explicitly prevents a `StratifiedShuffleSplit` API change from breaking the data pipeline without a visible change in `requirements.txt`.

---

### 11.5 Pillow for Inference Image Loading

Pillow (PIL) is required only for the inference pipeline. It is listed in `requirements.txt` because `torchvision` uses it internally for dataset loading anyway — it is not a new dependency, just one that must be pinned.

---

### 11.6 No TensorBoard Dependency in v1

TensorBoard (`tensorboard` package) is deliberately excluded from `requirements.txt`. The metrics pipeline writes CSV, which can be opened in any spreadsheet tool or visualized with matplotlib scripts. Adding TensorBoard would require `tensorboard` (and its transitive dependencies, including TensorFlow core on some systems) for what is essentially a live-view convenience. This will be reconsidered in a future version.

---

## 12. Future Improvements

These items are explicitly out of scope for v1 but are documented here so that future sessions do not have to rediscover them.

| Priority | Item | Notes |
|---|---|---|
| High | TensorBoard / W&B integration | Add `training.tensorboard: true` config flag; write `SummaryWriter` calls alongside CSV logging |
| High | ONNX export | `torch.onnx.export()` of best checkpoint; enables deployment to non-Python runtimes |
| Medium | `torch.compile()` for training speed | Wrap model with `torch.compile()` on PyTorch ≥2.0; significant speedup on CPU for repeated forward passes |
| Medium | Mixed-precision training (AMP) | `torch.cuda.amp.autocast()` reduces VRAM and speeds up GPU training; irrelevant for CPU-only runs |
| Medium | Learning rate warmup | Linear warmup for the first 1–2 epochs before cosine decay; improves stability with high initial LR |
| Medium | Custom dataset support | Abstract the data pipeline so it accepts any directory of labelled images, not only torchvision MNIST |
| Low | ResNet-style skip connections | Skip connections from input of a block to its output; would improve gradient flow and enable deeper architectures |
| Low | Fashion-MNIST extension | The pipeline is dataset-agnostic in all but the normalization constants and model architecture choices; easy to extend |
| Low | Distributed training (DDP) | `torch.nn.parallel.DistributedDataParallel`; not justified for MNIST but a useful learning exercise |
| Low | REST inference API | FastAPI wrapper around `Predictor`; useful if this becomes a demo application |
| Low | CI/CD pipeline | GitHub Actions workflow running `pytest tests/` on push; adds confidence that changes don't break acceptance criteria |

---

## 13. Key Commands

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# Install runtime dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v --cov=src

# Run tests for a specific module
pytest tests/test_cnn.py -v

# Train (auto-generates run_id)
python scripts/train.py --config config/default.yaml

# Train with explicit run_id
python scripts/train.py --config config/default.yaml --run-id cnn_experiment_01

# Resume training from checkpoint
python scripts/train.py --config config/default.yaml --resume checkpoints/cnn_20260518_143201/last.pt

# Evaluate a specific checkpoint on test set
python scripts/evaluate.py --config config/default.yaml --checkpoint checkpoints/cnn_20260518_143201/best.pt

# Infer on a single image
python scripts/infer.py --image path/to/digit.png --checkpoint checkpoints/cnn_20260518_143201/best.pt

# Infer on a directory of images
python scripts/infer.py --image-dir path/to/images/ --checkpoint checkpoints/cnn_20260518_143201/best.pt

# Generate plots for a completed run
python scripts/visualize.py --run-id cnn_20260518_143201

# Run hyperparameter tuning
python scripts/tune.py --config config/default.yaml

# Use MLP baseline instead of CNN
python scripts/train.py --config config/default.yaml  # after setting model.arch: mlp in config
```

---

## 14. Spec-Driven Process Rules

These rules govern how this project is developed. They are not guidelines — violating them invalidates the SDD methodology.

1. **No code before a spec section.** Every module must have a corresponding, approved section in `SPEC.md` before its first line of implementation code is written.

2. **Spec changes before implementation changes.** If a requirement changes during implementation (e.g., you realize the interface needs an extra argument), update `SPEC.md` first. Do not update the code and then update the spec to match.

3. **Acceptance criteria are executable.** Every `AC-XXX` entry in `SPEC.md §19` must have a corresponding test in `tests/`. A spec section is only marked `[x]` when its acceptance criteria pass under `pytest`.

4. **No magic numbers in source.** Every numeric constant that could reasonably vary between experiments belongs in `config/default.yaml`. If you find yourself writing `0.5` or `256` in source code, ask whether it belongs in the config.

5. **Tests use synthetic data.** No test shall download real MNIST data or depend on network access. Use `torch.randn` and `torch.randint` to create fixtures.

6. **Commit per completed phase.** Use the project-scoped `/git-commit` skill. Commit when a phase is complete and its acceptance criteria pass — not mid-implementation.

7. **Update this file when non-obvious decisions are made.** If you make a choice that a future reader might question, document it in §3–§6. The cost of writing two sentences now is far less than the cost of re-deriving the reasoning later.

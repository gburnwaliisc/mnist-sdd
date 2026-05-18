# MNIST Handwritten Digit Recognition — Claude Code Guide

## Project Overview

Spec-driven development of a dense neural network (DNN) for MNIST digit classification (0–9).
All pipelines — data loading, training/validation, inference, and hyperparameter tuning — are
controlled from a single YAML config file.

## Tech Stack

- **Language**: Python 3.10+
- **Framework**: PyTorch
- **Config**: YAML (via PyYAML)
- **Hyperparameter tuning**: Optuna
- **Logging**: Python `logging` module (structured, per-run log files)
- **Experiment tracking**: (TBD — see SPEC.md)

## Repository Layout

```
mnist-sdd/
├── CLAUDE.md
├── SPEC.md
├── config/
│   └── default.yaml          # master config (all pipelines)
├── src/
│   ├── data/
│   │   └── dataloader.py     # data loading & augmentation pipeline
│   ├── model/
│   │   └── network.py        # DNN architecture
│   ├── training/
│   │   └── trainer.py        # train + validation loop
│   ├── inference/
│   │   └── predictor.py      # single-image & batch inference
│   └── tuning/
│       └── tuner.py          # Optuna hyperparameter search
├── scripts/
│   ├── train.py              # CLI entry point: training
│   ├── evaluate.py           # CLI entry point: evaluation
│   ├── infer.py              # CLI entry point: inference
│   └── tune.py               # CLI entry point: hyperparameter tuning
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_trainer.py
│   └── test_predictor.py
├── checkpoints/              # saved model weights (git-ignored)
├── runs/                     # logs & tuning results (git-ignored)
└── requirements.txt
```

## Development Workflow

1. Agree on a spec section in `SPEC.md` before writing any implementation code.
2. Implement the module, then write tests for it.
3. Run tests with `pytest tests/` before marking a spec section complete.
4. All configurable values must live in `config/default.yaml`; no magic numbers in source.

## Config Contract

Every pipeline reads its settings from a single config dict loaded at startup.
The config is structured by pipeline:

```yaml
data: { ... }
model: { ... }
training: { ... }
inference: { ... }
tuning: { ... }
```

Modules must not import each other's config sections directly; they receive their
sub-dict as an argument.

## Coding Conventions

- Type annotations on all public functions and class methods.
- No global mutable state; pass config dicts explicitly.
- `torch.manual_seed` and `random.seed` set from config so runs are reproducible.
- Checkpoints saved as `checkpoints/{run_id}/epoch_{n}.pt`.
- All CLI scripts accept `--config` to override the default config path.

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python scripts/train.py --config config/default.yaml

# Evaluate on test set
python scripts/evaluate.py --config config/default.yaml --checkpoint checkpoints/<run_id>/best.pt

# Infer on a single image
python scripts/infer.py --image path/to/image.png --checkpoint checkpoints/<run_id>/best.pt

# Hyperparameter search
python scripts/tune.py --config config/default.yaml
```

## Spec-Driven Process

- `SPEC.md` is the source of truth for behaviour.
- Each spec section has a status tag: `[ ]` todo, `[~]` in-progress, `[x]` done.
- Do not implement anything not yet in the spec; propose additions to the spec first.

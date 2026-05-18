"""
MNIST data loading pipeline.

Provides a stratified, reproducible train/validation/test split of the MNIST
dataset and returns three ready-to-use DataLoader objects.  All behaviour is
controlled through the config dict — no hard-coded constants appear here.

Public API
----------
get_dataloaders(cfg)       -> tuple[DataLoader, DataLoader, DataLoader]
build_transforms(cfg, ...) -> transforms.Compose

SPEC reference: SDD-MNIST-001 §6 (Dataset), §1 (FR-DAT-001 … FR-DAT-010)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transform builder
# ---------------------------------------------------------------------------

def build_transforms(cfg: dict, *, augment: bool = False) -> transforms.Compose:
    """
    Build the preprocessing transform pipeline for one data split.

    The pipeline is:
        [RandomAffine]   — optional, training split only
        ToTensor         — PIL uint8 → float32 in [0, 1]
        Normalize        — channel-wise (mean, std) from config
        [Flatten]        — only when model.arch == 'mlp'

    Args:
        cfg:     Full config dict.  Reads cfg['data']['normalize'],
                 cfg['data']['augmentation'], and cfg['model']['arch'].
        augment: When True *and* cfg['data']['augmentation'] is True,
                 a RandomAffine transform is prepended.  Should be True
                 only for the training split.

    Returns:
        A composed torchvision transform pipeline.
    """
    data_cfg = cfg["data"]
    arch     = cfg["model"]["arch"]
    mean     = data_cfg["normalize"]["mean"]
    std      = data_cfg["normalize"]["std"]

    pipeline: list = []

    # 1. Optional augmentation -----------------------------------------------
    #    Guarded by both the caller flag (augment=True) and the config switch
    #    (data.augmentation: true).  Either alone is not sufficient —  the
    #    caller flag prevents augmentation leaking into val/test even if the
    #    config switch is on, and the config switch lets the user disable
    #    augmentation without touching code.
    if augment and data_cfg.get("augmentation", False):
        pipeline.append(
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
        )
        logger.debug("Augmentation: RandomAffine(degrees=10, translate=(0.1,0.1))")

    # 2. PIL → float tensor in [0, 1] ----------------------------------------
    pipeline.append(transforms.ToTensor())

    # 3. Channel-wise normalisation ------------------------------------------
    #    mean and std are MNIST dataset statistics (0.1307, 0.3081).
    #    They are treated as constants from config rather than being
    #    recomputed at runtime.  See CLAUDE.md §7.2 for the rationale.
    pipeline.append(transforms.Normalize(mean=[mean], std=[std]))

    # 4. Architecture-specific flatten ---------------------------------------
    #    The CNN forward pass expects (B, 1, 28, 28).
    #    The MLP forward pass expects (B, 784).
    #    Flattening here rather than inside the model's forward() keeps
    #    both model classes free of input-shape branching and lets data
    #    pipeline tests assert the correct shape without a model.
    #    transforms.Lambda is picklable in PyTorch >= 1.8 (we require 2.1+).
    if arch == "mlp":
        pipeline.append(transforms.Lambda(lambda x: x.view(-1)))
        logger.debug("Appended flatten transform for MLP architecture")

    return transforms.Compose(pipeline)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _stratified_split(
    labels: np.ndarray,
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Return (train_indices, val_indices) with guaranteed class stratification.

    Uses sklearn.model_selection.StratifiedShuffleSplit which provides a
    single random stratified split.  torch.utils.data.random_split has no
    stratification support; see CLAUDE.md §4.4 for the choice rationale.

    Args:
        labels:    Integer class label for every sample (shape: N,).
        val_split: Fraction of samples for validation; must be in (0, 1).
        seed:      Controls the split — identical seed → identical indices.

    Returns:
        train_indices: List of integer indices into the source dataset.
        val_indices:   List of integer indices into the source dataset.
    """
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_split,
        random_state=seed,
    )
    all_indices = np.arange(len(labels))
    train_idx, val_idx = next(sss.split(all_indices, labels))
    return train_idx.tolist(), val_idx.tolist()


# ---------------------------------------------------------------------------
# Worker seed initialiser
# ---------------------------------------------------------------------------

def _worker_init_fn(worker_id: int, base_seed: int) -> None:
    """
    Seed every DataLoader worker process deterministically.

    Why base_seed + worker_id:
        All workers seeded identically (fork behaviour on Linux) would
        produce correlated random sequences — e.g. augmentation operations
        in worker 0 and worker 1 would apply the same transforms to their
        respective batches.  Unique seeds per worker break this correlation
        while keeping the overall data ordering reproducible.

    Args:
        worker_id: Zero-based index assigned by PyTorch to each worker.
        base_seed: Derived from cfg['seed']; closed over by the caller.
    """
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader).

    Downloads MNIST to cfg['data']['root'] on the first call if the
    dataset files are absent.  Produces a reproducible, stratified
    train/validation split from the official 60,000-sample training set.
    The 10,000-sample test set is kept as a true holdout and is never
    used during training or hyperparameter selection.

    Split sizes (default val_split=0.1):
        train:  54,000 samples  (shuffled each epoch)
        val:     6,000 samples  (fixed order)
        test:   10,000 samples  (fixed order, holdout)

    Args:
        cfg: Full config dict.  Consumed keys:
            cfg['seed']                  — global RNG seed
            cfg['data']['root']          — dataset download directory
            cfg['data']['batch_size']    — samples per batch
            cfg['data']['num_workers']   — DataLoader subprocess count
            cfg['data']['pin_memory']    — pin host tensors to CUDA memory
            cfg['data']['val_split']     — fraction of train set for val
            cfg['data']['augmentation']  — enable training augmentation
            cfg['data']['normalize']     — {'mean': float, 'std': float}
            cfg['model']['arch']         — 'cnn' | 'mlp'

    Returns:
        train_loader: Shuffled.  Augmentation applied if configured.
        val_loader:   Unshuffled.  No augmentation.
        test_loader:  Unshuffled.  No augmentation.

    Raises:
        ValueError:   If val_split is not strictly in (0, 1).
        RuntimeError: If MNIST cannot be downloaded or found on disk.
    """
    data_cfg  = cfg["data"]
    seed      = cfg["seed"]
    val_split = data_cfg["val_split"]
    root      = Path(data_cfg["root"])

    # --- Input validation ---------------------------------------------------
    if not (0.0 < val_split < 1.0):
        raise ValueError(
            f"data.val_split must be in the open interval (0, 1); "
            f"got {val_split!r}."
        )

    root.mkdir(parents=True, exist_ok=True)

    # --- Download / locate dataset ------------------------------------------
    logger.info("Loading MNIST dataset from '%s'", root)
    try:
        # Load without transforms to access .targets for stratification.
        # MNIST.targets is a pre-loaded tensor — accessing it is O(1).
        raw_train = datasets.MNIST(
            root=str(root), train=True, download=True, transform=None
        )
        # Trigger download of test files even though we don't use raw_test here.
        datasets.MNIST(root=str(root), train=False, download=True, transform=None)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MNIST from '{root}'. "
            "Ensure network access is available for the first download, or "
            "that the four MNIST binary files already exist under "
            f"'{root / 'MNIST' / 'raw'}'.\n"
            f"Underlying error: {exc}"
        ) from exc

    # --- Stratified split ---------------------------------------------------
    labels = raw_train.targets.numpy()          # (60000,) int array
    train_idx, val_idx = _stratified_split(labels, val_split, seed)

    logger.info(
        "Split sizes — train: %d | val: %d | test: 10000",
        len(train_idx),
        len(val_idx),
    )
    _log_class_distribution("train", labels[np.array(train_idx)])
    _log_class_distribution("val",   labels[np.array(val_idx)])

    # --- Build transform pipelines ------------------------------------------
    train_tf = build_transforms(cfg, augment=True)
    eval_tf  = build_transforms(cfg, augment=False)

    # Re-load the training partition twice: once with training transforms
    # and once with evaluation transforms.  Both Dataset objects are lazy —
    # torchvision decodes images per __getitem__ call, not at construction.
    # The two objects share the same on-disk files; there is no data duplication.
    # Subset then restricts each to the correct index partition.
    train_full = datasets.MNIST(
        root=str(root), train=True, download=False, transform=train_tf
    )
    eval_full = datasets.MNIST(
        root=str(root), train=True, download=False, transform=eval_tf
    )
    test_ds = datasets.MNIST(
        root=str(root), train=False, download=False, transform=eval_tf
    )

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(eval_full,  val_idx)

    # --- DataLoader configuration -------------------------------------------
    batch_size  = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    pin_memory  = data_cfg.get("pin_memory", True)

    # An explicit Generator isolates shuffle randomness from the global RNG
    # state.  Without it, any random operation between torch.manual_seed()
    # and the first DataLoader iteration alters the shuffle permutation.
    # See CLAUDE.md §5 for the full rationale.
    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init(worker_id: int) -> None:
        _worker_init_fn(worker_id, base_seed=seed)

    # persistent_workers keeps subprocess pool alive between epochs,
    # avoiding spawn overhead (≈ O(seconds) per epoch on some systems).
    # Guarded by num_workers > 0: the option has no effect with in-process
    # loading and emits a UserWarning if set with num_workers=0.
    persist = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        generator=generator,
        persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        persistent_workers=persist,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        persistent_workers=persist,
    )

    logger.info(
        "DataLoaders ready — batch_size: %d | num_workers: %d | pin_memory: %s | arch: %s",
        batch_size,
        num_workers,
        pin_memory,
        cfg["model"]["arch"],
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _log_class_distribution(split_name: str, labels: np.ndarray) -> None:
    """Log per-class sample counts at DEBUG level (not shown in INFO runs)."""
    counts = np.bincount(labels, minlength=10)
    dist   = "  ".join(f"{i}:{c}" for i, c in enumerate(counts))
    logger.debug("Class distribution [%s]  %s", split_name, dist)

"""
Metrics computation for MNIST training and evaluation.

All functions accept CPU or GPU tensors and return Python scalars or
numpy arrays.  They are pure functions — no side effects, no state.

SPEC reference: SDD-MNIST-001 §9
"""

from __future__ import annotations

import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Fraction of samples whose argmax prediction matches the true label.

    Args:
        logits: (B, C) raw model output.
        labels: (B,)  integer class indices in [0, C).

    Returns:
        Accuracy in [0, 1].
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_per_class_accuracy(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    num_classes: int = 10,
) -> list[float]:
    """
    Per-class accuracy: correct_k / total_k for each class k.

    Args:
        all_preds:   (N,) predicted class indices.
        all_labels:  (N,) true class indices.
        num_classes: Number of classes.

    Returns:
        List of length num_classes. Entry k is accuracy for class k.
        If a class has no samples, its accuracy is 0.0.
    """
    per_class = []
    for k in range(num_classes):
        mask = all_labels == k
        total = mask.sum().item()
        if total == 0:
            per_class.append(0.0)
        else:
            correct = (all_preds[mask] == k).sum().item()
            per_class.append(correct / total)
    return per_class


def compute_confusion_matrix(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    num_classes: int = 10,
) -> torch.Tensor:
    """
    Return a (num_classes, num_classes) confusion matrix.

    C[i, j] = number of samples with true label i predicted as j.

    Uses bincount with a compound index to avoid a Python-level double
    loop over all predictions.

    Args:
        all_preds:   (N,) predicted class indices.
        all_labels:  (N,) true class indices.
        num_classes: Number of classes.

    Returns:
        LongTensor of shape (num_classes, num_classes).
    """
    idx = all_labels * num_classes + all_preds
    cm  = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def compute_classification_report(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    num_classes: int = 10,
) -> dict:
    """
    Per-class and macro-averaged precision, recall, F1, and support.

    All quantities are derived from the confusion matrix — no sklearn
    dependency is required.

    Precision_k = C[k,k] / C[:,k].sum()   (column k = all predicted-k)
    Recall_k    = C[k,k] / C[k,:].sum()   (row    k = all true-k)
    F1_k        = 2 · P_k · R_k / (P_k + R_k)
    Zero denominators produce 0.0 (not NaN or ZeroDivisionError).

    Args:
        all_preds:   (N,) predicted class indices.
        all_labels:  (N,) true class indices.
        num_classes: Number of classes.

    Returns:
        dict with keys:
            'per_class' : list of dicts, one per class:
                          {'class', 'precision', 'recall', 'f1', 'support'}
            'macro'     : dict {'precision', 'recall', 'f1'}
            'accuracy'  : float — overall accuracy
            'n_samples' : int — total samples evaluated
    """
    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)
    n_samples = len(all_labels)
    accuracy  = (all_preds == all_labels).float().mean().item()

    per_class = []
    for k in range(num_classes):
        tp = cm[k, k].item()
        fp = cm[:, k].sum().item() - tp   # predicted k but not true k
        fn = cm[k, :].sum().item() - tp   # true k but not predicted k
        support = cm[k, :].sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom     = precision + recall
        f1        = 2 * precision * recall / denom if denom > 0 else 0.0

        per_class.append({
            "class":     k,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "support":   int(support),
        })

    macro_precision = sum(c["precision"] for c in per_class) / num_classes
    macro_recall    = sum(c["recall"]    for c in per_class) / num_classes
    macro_f1        = sum(c["f1"]        for c in per_class) / num_classes

    return {
        "per_class": per_class,
        "macro": {
            "precision": macro_precision,
            "recall":    macro_recall,
            "f1":        macro_f1,
        },
        "accuracy":  accuracy,
        "n_samples": n_samples,
    }


def compute_macro_f1(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    num_classes: int = 10,
) -> float:
    """
    Unweighted mean of per-class F1 scores (macro-average F1).

    F1_k = 2 * precision_k * recall_k / (precision_k + recall_k)
    where zero-denominator terms are set to 0.

    Args:
        all_preds:   (N,) predicted class indices.
        all_labels:  (N,) true class indices.
        num_classes: Number of classes.

    Returns:
        Macro F1 in [0, 1].
    """
    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)
    f1_scores = []
    for k in range(num_classes):
        tp = cm[k, k].item()
        fp = cm[:, k].sum().item() - tp
        fn = cm[k, :].sum().item() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom     = precision + recall
        f1        = 2 * precision * recall / denom if denom > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / num_classes

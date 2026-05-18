"""
Evaluator — standalone post-training evaluation against the MNIST test set.

Design:
    The Evaluator is fully self-contained: it reads the checkpoint's embedded
    config dict (not the current default.yaml) to reconstruct the model.
    This guarantees that a checkpoint from a run with non-default
    conv_channels or fc_hidden is evaluated with the architecture it was
    trained with.

Outputs written to runs/{run_id}/:
    confusion_matrix.png  — raw-count and row-normalised heatmaps side-by-side
    test_results.json     — all metrics in machine-readable form

Console output:
    ASCII table with per-class and macro precision, recall, F1, support.
    No ANSI colour codes — compatible with all terminals and CI log viewers.

SPEC reference: SDD-MNIST-001 §8.6, §9.2, §10.1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import build_model
from src.training.metrics import compute_classification_report, compute_confusion_matrix

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Load a checkpoint and run full evaluation on the MNIST test set.

    Args:
        checkpoint_path: Path to a .pt checkpoint written by Trainer.
        run_dir:         Directory for output files.  Defaults to
                         runs/{checkpoint_parent_name}/ so outputs sit
                         alongside training logs for the same run.
        batch_size:      Override the checkpoint config's batch size.
                         Useful for evaluating on memory-constrained machines.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        run_dir: str | Path | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.ckpt_path = Path(checkpoint_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint — weights_only=False because the dict contains
        # non-tensor objects (config dict, metrics dict).  Source is trusted.
        self.ckpt = torch.load(
            self.ckpt_path, map_location=self.device, weights_only=False
        )
        self.cfg = self.ckpt["config"]

        # Allow batch_size override without mutating the embedded config
        if batch_size is not None:
            import copy
            self.cfg = copy.deepcopy(self.cfg)
            self.cfg["data"]["batch_size"] = batch_size

        # Infer run_id from the checkpoint's parent directory name
        run_id = self.ckpt_path.parent.name
        if run_dir is None:
            log_dir = Path(self.cfg["training"].get("log_dir", "./runs"))
            self.run_dir = log_dir / run_id
        else:
            self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Reconstruct model from the checkpoint's embedded config
        self.model = build_model(self.cfg)
        self.model.load_state_dict(self.ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Evaluator: checkpoint=%s  epoch=%d  device=%s",
            self.ckpt_path, self.ckpt.get("epoch", "?"), self.device,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run full evaluation on the MNIST test set.

        Returns:
            dict containing test_loss, test_acc, macro precision/recall/F1,
            per_class metrics, confusion_matrix (nested list), n_samples.

        Side effects:
            Writes confusion_matrix.png and test_results.json to run_dir.
            Prints evaluation report to stdout.
        """
        test_loader = self._build_test_loader()
        raw_results = self._eval_loop(test_loader)

        self._save_confusion_matrix(raw_results["cm"])
        self._save_json(raw_results["report"], raw_results["test_loss"])
        self._print_report(raw_results["report"], raw_results["test_loss"])

        # Compose the full return dict
        report = raw_results["report"]
        return {
            "test_loss":        raw_results["test_loss"],
            "test_acc":         report["accuracy"],
            "n_samples":        report["n_samples"],
            "per_class":        report["per_class"],
            "macro":            report["macro"],
            "confusion_matrix": raw_results["cm"].tolist(),
        }

    # ── Private helpers ─────────────────────────────────────────────────────

    def _build_test_loader(self) -> DataLoader:
        from src.data.dataset import get_dataloaders
        _, _, test_loader = get_dataloaders(self.cfg)
        return test_loader

    def _eval_loop(self, loader: DataLoader) -> dict:
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0
        all_preds:  list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss   = criterion(logits, labels)
                bs     = labels.size(0)

                total_loss    += loss.item() * bs
                total_samples += bs
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())

        preds_t  = torch.cat(all_preds)
        labels_t = torch.cat(all_labels)
        n_classes = self.cfg["model"]["num_classes"]

        return {
            "test_loss": total_loss / total_samples,
            "cm":        compute_confusion_matrix(preds_t, labels_t, n_classes),
            "report":    compute_classification_report(preds_t, labels_t, n_classes),
        }

    def _save_confusion_matrix(self, cm: torch.Tensor) -> None:
        """Save raw-count and row-normalised confusion matrices side-by-side."""
        cm_np  = cm.numpy().astype(float)
        # Row-normalise: each row sum → 1.0  (diagonal = recall per class)
        row_sums = cm_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1   # avoid divide-by-zero for empty classes
        cm_norm = cm_np / row_sums

        n = cm_np.shape[0]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, data, title, fmt in [
            (axes[0], cm_np,  "Confusion Matrix — Raw Counts",        ".0f"),
            (axes[1], cm_norm, "Confusion Matrix — Row Normalised (%)", ".1%"),
        ]:
            im = ax.imshow(data, cmap="Blues", aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(range(n))
            ax.set_yticklabels(range(n))
            ax.set_xlabel("Predicted digit")
            ax.set_ylabel("True digit")
            ax.set_title(title)
            # Annotate cells
            thresh = data.max() / 2.0
            for i in range(n):
                for j in range(n):
                    ax.text(
                        j, i, format(data[i, j], fmt),
                        ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > thresh else "black",
                    )

        epoch = self.ckpt.get("epoch", "?")
        fig.suptitle(
            f"Checkpoint: {self.ckpt_path.name}  |  Epoch: {epoch}",
            fontsize=11,
        )
        fig.tight_layout()
        out = self.run_dir / "confusion_matrix.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", out)

    def _save_json(self, report: dict, test_loss: float) -> None:
        payload = {
            "checkpoint":  str(self.ckpt_path),
            "epoch":       self.ckpt.get("epoch"),
            "test_loss":   test_loss,
            "test_acc":    report["accuracy"],
            "n_samples":   report["n_samples"],
            "macro":       report["macro"],
            "per_class":   report["per_class"],
        }
        out = self.run_dir / "test_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Test results saved to %s", out)

    def _print_report(self, report: dict, test_loss: float) -> None:
        width = 62
        border = "=" * width

        def _row(digit, p, r, f, sup):
            return (
                f"  {digit:^5}  {p:>9.2%}  {r:>9.2%}  {f:>9.2%}  {sup:>9,}"
            )

        lines = [
            "",
            border,
            f"  MNIST Evaluation Report",
            f"  Checkpoint : {self.ckpt_path}",
            f"  Epoch      : {self.ckpt.get('epoch', 'unknown')}",
            border,
            f"  Samples tested : {report['n_samples']:>10,}",
            f"  Test loss      : {test_loss:>10.4f}",
            f"  Test accuracy  : {report['accuracy']:>10.2%}",
            f"  Macro F1       : {report['macro']['f1']:>10.4f}",
            "",
            f"  {'Digit':^5}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}  {'Support':>9}",
            "  " + "-" * 57,
        ]

        for c in report["per_class"]:
            lines.append(_row(
                c["class"], c["precision"], c["recall"], c["f1"], c["support"]
            ))

        m = report["macro"]
        lines += [
            "  " + "-" * 57,
            _row("Macro", m["precision"], m["recall"], m["f1"], report["n_samples"]),
            border,
            f"  Confusion matrix : {self.run_dir / 'confusion_matrix.png'}",
            f"  Full results     : {self.run_dir / 'test_results.json'}",
            border,
            "",
        ]

        print("\n".join(lines))

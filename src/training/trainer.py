"""
Trainer — full training, validation, and test evaluation pipeline.

Public interface (SPEC §8.7):
    Trainer(model, cfg, run_id)
    Trainer.fit(train_loader, val_loader)  → dict of final epoch metrics
    Trainer.evaluate(test_loader)          → dict of all test metrics

Logging sinks (SPEC §10.1):
    stdout           : epoch summary every epoch
    runs/{id}/run.log: Python logging (INFO)
    runs/{id}/metrics.csv : per-epoch metrics (epoch, train_loss, val_loss, val_acc, lr)
    runs/{id}/config.yaml : config snapshot (written once at fit() entry)
    runs/{id}/test_results.json : test metrics (written once after evaluate())
    checkpoints/{id}/best.pt   : best monitored metric
    checkpoints/{id}/last.pt   : end of every epoch
    checkpoints/{id}/epoch_{n:03d}.pt : optional per-epoch

Matplotlib Agg backend is selected here before pyplot is imported so that
plot generation works on headless machines (no display server required).

SPEC reference: SDD-MNIST-001 §8, §9, §10
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.training.early_stopping import EarlyStopping
from src.training.metrics import (
    compute_confusion_matrix,
    compute_macro_f1,
    compute_per_class_accuracy,
)
from src.utils import (
    AverageMeter,
    Timer,
    load_checkpoint,
    plot_training_curves,
    save_checkpoint,
    save_json,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_optimizer(
    model: nn.Module,
    cfg_training: dict,
) -> torch.optim.Optimizer:
    name = cfg_training["optimizer"].lower()
    lr   = cfg_training["learning_rate"]
    wd   = cfg_training["weight_decay"]

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=cfg_training["momentum"],
        )
    else:
        raise ValueError(f"Unsupported optimizer '{name}'. Valid: 'adam', 'sgd'.")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg_training: dict,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    sched_cfg = cfg_training.get("lr_scheduler", {})
    sched_type = sched_cfg.get("type", "none").lower()

    if sched_type == "none":
        return None
    elif sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg["step_size"],
            gamma=sched_cfg.get("gamma", 0.1),
        )
    elif sched_type == "cosine":
        t_max = sched_cfg.get("T_max") or cfg_training["epochs"]
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=sched_cfg.get("factor", 0.1),
            patience=sched_cfg.get("patience", 5),
        )
    else:
        raise ValueError(
            f"Unsupported lr_scheduler '{sched_type}'. "
            "Valid: 'none', 'step', 'cosine', 'plateau'."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Orchestrates the full training, validation, and test evaluation loop.

    Args:
        model:  Instantiated nn.Module (MNISTConvNet or MNISTNet).
        cfg:    Full config dict.  Trainer reads cfg['training'].
        run_id: Unique identifier for this run.
                Convention: '{arch}_{YYYYMMDD_HHMMSS}'.
    """

    def __init__(self, model: nn.Module, cfg: dict, run_id: str) -> None:
        self.model  = model
        self.cfg    = cfg
        self.run_id = run_id

        t = cfg["training"]
        self.epochs         = t["epochs"]
        self.cfg_training   = t
        self.auto_visualize = t.get("auto_visualize", False)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info("Trainer: device=%s  run_id=%s", self.device, run_id)

        # Directories
        self.run_dir  = Path(t.get("log_dir", "./runs")) / run_id
        self.ckpt_dir = Path(t["checkpoint"]["dir"]) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # File sinks
        self.csv_path  = self.run_dir / "metrics.csv"
        self.log_path  = self.run_dir / "run.log"
        self.plot_path = self.run_dir / "training_curves.png"

        self._setup_file_logger()

        # Optimizer and scheduler (created fresh — not restored here)
        self.optimizer = _build_optimizer(self.model, t)
        self.scheduler = _build_scheduler(self.optimizer, t)

        # Early stopping
        es_cfg = t.get("early_stopping", {})
        if es_cfg.get("enabled", True):
            self.early_stopping: Optional[EarlyStopping] = EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=es_cfg.get("patience", 5),
                min_delta=es_cfg.get("min_delta", 0.0),
            )
        else:
            self.early_stopping = None

        # State
        self._epoch_metrics: list[dict] = []
        self._best_metric_value: float  = (
            float("inf")
            if (self.early_stopping is None or self.early_stopping.monitor == "val_loss")
            else float("-inf")
        )
        self._monitor = (
            self.early_stopping.monitor if self.early_stopping is not None
            else "val_loss"
        )

        # Log model summary
        if hasattr(model, "count_parameters"):
            stats = model.count_parameters()
            logger.info(
                "Model: total_params=%s  trainable_params=%s",
                f"{stats['total']:,}",
                f"{stats['trainable']:,}",
            )

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict:
        """
        Run the full training loop.

        Returns the metrics dict from the final completed epoch.
        Saves checkpoints and logs metrics each epoch.
        """
        self._save_config_snapshot()

        # Optional resume (restore state from last.pt)
        start_epoch = self._maybe_resume()

        final_metrics: dict = {}

        for epoch in range(start_epoch, self.epochs + 1):
            with Timer() as epoch_timer:
                self.model.train()
                train_loss = self._train_epoch(train_loader)

                self.model.eval()
                val_loss, val_acc = self._validate(val_loader)

            lr      = self.optimizer.param_groups[0]["lr"]
            elapsed = epoch_timer.elapsed

            metrics = {
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "lr":         lr,
            }
            self._epoch_metrics.append(metrics)
            final_metrics = metrics

            self._log_epoch(metrics, elapsed)
            self._append_csv(metrics)
            self._save_checkpoint("last.pt", epoch, metrics)

            # Save best checkpoint
            monitored = metrics[self._monitor]
            if self._is_improvement(monitored):
                self._best_metric_value = monitored
                self._save_checkpoint("best.pt", epoch, metrics)
                logger.info("Saved best checkpoint (epoch %d, %s=%.6f)", epoch, self._monitor, monitored)

            # Optional per-epoch checkpoint
            if self.cfg_training["checkpoint"].get("save_every_epoch", False):
                self._save_checkpoint(f"epoch_{epoch:03d}.pt", epoch, metrics)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping
            if self.early_stopping is not None:
                stop = self.early_stopping.step(metrics[self._monitor])
                if stop:
                    logger.info("Early stopping triggered at epoch %d.", epoch)
                    break

        if self.auto_visualize:
            self._plot_training_curves()

        return final_metrics

    def evaluate(self, test_loader: DataLoader) -> dict:
        """
        Load best.pt and evaluate on test_loader.

        Returns a dict with test_loss, test_acc, per_class_acc,
        confusion_matrix (as nested list), and macro_f1.
        Saves results to runs/{run_id}/test_results.json.
        """
        best_path = self.ckpt_dir / "best.pt"
        if best_path.exists():
            ckpt = load_checkpoint(best_path, device=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            logger.info("Loaded best.pt from epoch %d for test evaluation.", ckpt["epoch"])
        else:
            logger.warning("best.pt not found; evaluating with current model weights.")

        self.model.eval()
        criterion  = nn.CrossEntropyLoss()
        loss_meter = AverageMeter()
        all_preds:  list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                bs     = labels.size(0)

                loss_meter.update(criterion(logits, labels).item(), n=bs)
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())

        self.model.train()

        all_preds_t  = torch.cat(all_preds)
        all_labels_t = torch.cat(all_labels)

        num_classes = self.cfg["model"]["num_classes"]
        test_loss   = loss_meter.avg
        test_acc    = (all_preds_t == all_labels_t).float().mean().item()
        per_cls_acc = compute_per_class_accuracy(all_preds_t, all_labels_t, num_classes)
        cm          = compute_confusion_matrix(all_preds_t, all_labels_t, num_classes)
        macro_f1    = compute_macro_f1(all_preds_t, all_labels_t, num_classes)

        results = {
            "test_loss":        test_loss,
            "test_acc":         test_acc,
            "per_class_acc":    per_cls_acc,
            "confusion_matrix": cm.tolist(),
            "macro_f1":         macro_f1,
        }

        logger.info(
            "Test evaluation: loss=%.4f  acc=%.4f  macro_f1=%.4f",
            test_loss, test_acc, macro_f1,
        )

        save_json(results, self.run_dir / "test_results.json")
        logger.info("Test results saved to %s", self.run_dir / "test_results.json")

        return results

    # ── Private helpers ─────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> float:
        criterion  = nn.CrossEntropyLoss()
        loss_meter = AverageMeter()

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            bs     = labels.size(0)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), n=bs)

        return loss_meter.avg

    def _validate(self, loader: DataLoader) -> tuple[float, float]:
        criterion    = nn.CrossEntropyLoss()
        loss_meter   = AverageMeter()
        correct_meter = AverageMeter()

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                bs     = labels.size(0)

                loss_meter.update(criterion(logits, labels).item(), n=bs)
                # Pass batch-accuracy fraction so AverageMeter accumulates
                # total_correct / total_samples correctly across variable-size batches.
                correct_meter.update(
                    (logits.argmax(dim=1) == labels).float().mean().item(), n=bs
                )

        return loss_meter.avg, correct_meter.avg

    def _is_improvement(self, value: float) -> bool:
        if self._monitor == "val_loss":
            return value < self._best_metric_value
        else:
            return value > self._best_metric_value

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict) -> None:
        save_checkpoint(
            payload={
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
                "metrics":         metrics,
                "config":          self.cfg,
            },
            path=self.ckpt_dir / filename,
        )

    def _log_epoch(self, metrics: dict, elapsed: float) -> None:
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f  val_loss=%.4f  "
            "val_acc=%.4f  lr=%.2e  time=%.1fs",
            metrics["epoch"], self.epochs,
            metrics["train_loss"], metrics["val_loss"],
            metrics["val_acc"], metrics["lr"],
            elapsed,
        )

    def _append_csv(self, metrics: dict) -> None:
        import csv

        columns = ["epoch", "train_loss", "val_loss", "val_acc", "lr"]
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

    def _plot_training_curves(self) -> None:
        plot_training_curves(
            self._epoch_metrics,
            output_path=self.plot_path,
            title=self.run_id,
        )

    def _save_config_snapshot(self) -> None:
        cfg_path = self.run_dir / "config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(self.cfg, f, default_flow_style=False)
        logger.info("Config snapshot saved to %s", cfg_path)

    def _setup_file_logger(self) -> None:
        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s  %(message)s")
        )
        logging.getLogger().addHandler(fh)

    def _maybe_resume(self) -> int:
        """
        If cfg['training']['resume'] points to a checkpoint, restore state
        and return the epoch to resume from.  Otherwise return 1.
        """
        resume_path = self.cfg_training.get("resume")
        if not resume_path:
            return 1

        path = Path(resume_path)
        if not path.exists():
            logger.warning("Resume checkpoint not found: %s — starting from epoch 1.", path)
            return 1

        ckpt = load_checkpoint(path, device=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler and ckpt.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])

        resume_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from %s — continuing from epoch %d.", path, resume_epoch)
        return resume_epoch

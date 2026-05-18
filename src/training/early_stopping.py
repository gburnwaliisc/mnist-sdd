"""
EarlyStopping — halts training when a monitored metric stops improving.

Improvement direction:
    val_loss → improvement means decrease by more than min_delta.
    val_acc  → improvement means increase by more than min_delta.

Sentinel initialisation (see CLAUDE.md §10.4):
    best_value starts at +inf for val_loss and -inf for val_acc so the
    first epoch always registers as an improvement.

SPEC reference: SDD-MNIST-001 §8.5
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Track a monitored metric across epochs and signal when to stop.

    Args:
        monitor:   Metric name — 'val_loss' (lower is better) or
                   'val_acc' (higher is better).
        patience:  Number of epochs without improvement before stopping.
        min_delta: Minimum change that counts as an improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
    ) -> None:
        if monitor not in ("val_loss", "val_acc"):
            raise ValueError(
                f"monitor must be 'val_loss' or 'val_acc'; got {monitor!r}"
            )
        if patience < 1:
            raise ValueError(f"patience must be >= 1; got {patience!r}")
        if min_delta < 0.0:
            raise ValueError(f"min_delta must be >= 0; got {min_delta!r}")

        self.monitor   = monitor
        self.patience  = patience
        self.min_delta = min_delta

        # Sentinel: first epoch always improves.
        self.best_value: float = float("inf") if monitor == "val_loss" else float("-inf")
        self.counter:    int   = 0
        self.should_stop: bool = False

    # ------------------------------------------------------------------

    def step(self, value: float) -> bool:
        """
        Update state with the latest metric value.

        Returns:
            True if training should stop, False otherwise.
        """
        improved = self._is_improvement(value)

        if improved:
            self.best_value = value
            self.counter = 0
            logger.debug(
                "EarlyStopping: %s improved to %.6f", self.monitor, value
            )
        else:
            self.counter += 1
            logger.debug(
                "EarlyStopping: no improvement (%d / %d)",
                self.counter, self.patience,
            )

        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(
                "EarlyStopping: patience %d exhausted on %s (best=%.6f). "
                "Stopping training.",
                self.patience, self.monitor, self.best_value,
            )

        return self.should_stop

    # ------------------------------------------------------------------

    def _is_improvement(self, value: float) -> bool:
        if self.monitor == "val_loss":
            return value < self.best_value - self.min_delta
        else:  # val_acc
            return value > self.best_value + self.min_delta

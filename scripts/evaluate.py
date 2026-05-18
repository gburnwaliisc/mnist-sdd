"""
scripts/evaluate.py — CLI entry point for post-training evaluation.

Usage examples:
    python scripts/evaluate.py --checkpoint checkpoints/cnn_20260518/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/cnn_20260518/best.pt \\
        --run-dir runs/my_eval --batch-size 256

The script loads the checkpoint's embedded config to reconstruct the model,
downloads MNIST if necessary, runs the test-set evaluation, and writes:
    runs/{run_id}/confusion_matrix.png
    runs/{run_id}/test_results.json

SPEC reference: SDD-MNIST-001 §8.6
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MNIST checkpoint on the test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to the .pt checkpoint file (e.g. checkpoints/run_id/best.pt).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory for output files (confusion_matrix.png, test_results.json). "
            "Defaults to runs/{checkpoint_parent_name}/."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Override the checkpoint config batch size for the test DataLoader.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import here (after basicConfig) so module-level loggers pick up the config
    from src.evaluation.evaluator import Evaluator

    try:
        evaluator = Evaluator(
            checkpoint_path=args.checkpoint,
            run_dir=args.run_dir,
            batch_size=args.batch_size,
        )
        results = evaluator.run()
        sys.exit(0)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logging.exception("Evaluation failed: %s", exc)
        sys.exit(2)


# Guard required on Windows: DataLoader workers re-import the script,
# causing infinite recursion without this check (CLAUDE.md §10.2).
if __name__ == "__main__":
    main()

import torch.nn as nn

from src.model.cnn import MNISTConvNet

__all__ = ["MNISTConvNet", "build_model"]


def build_model(cfg: dict) -> nn.Module:
    """
    Instantiate the model specified by cfg['model']['arch'].

    Args:
        cfg: Full config dict.

    Returns:
        Untrained nn.Module ready for parameter loading or training.

    Raises:
        ValueError: If arch is not recognised.
        NotImplementedError: If arch is recognised but not yet implemented.
    """
    arch = cfg["model"]["arch"]
    if arch == "cnn":
        return MNISTConvNet(cfg)
    elif arch == "mlp":
        raise NotImplementedError(
            "MNISTNet (MLP baseline) is not yet implemented. "
            "Set model.arch: cnn or wait for Phase 2."
        )
    else:
        raise ValueError(
            f"Unknown model arch '{arch}'. Valid options: 'cnn', 'mlp'."
        )

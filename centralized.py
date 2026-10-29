"""Centralized (non-federated) baselines trained with the exact same model and optimiser."""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from dataset import ImageTensorDataset
from model import build_model, evaluate_detailed, train

logger = logging.getLogger(__name__)


def train_baseline(
    name: str,
    train_dataset: ImageTensorDataset,
    test_dataset: ImageTensorDataset,
    in_channels: int,
    num_classes: int,
    num_rounds: int,
    epochs_per_round: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> list[dict[str, float]]:
    """Train on `train_dataset` and evaluate on the test set every `epochs_per_round` epochs.

    Using the same initial weights (same seed) and the same number of epochs per "round" as the
    federated run means one FedAvg round and one baseline round use the same compute budget
    over the pooled data.
    Returns one record per round, with round 0 being the untrained initial model.
    """
    model = build_model(in_channels, num_classes, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)

    history: list[dict[str, float]] = []
    m = evaluate_detailed(model, test_loader, device)
    history.append({"round": 0, **m, "train_loss": float("nan"), "train_accuracy": float("nan")})
    logger.info("%s | round  0 | test loss=%.4f acc=%.4f", name, m["loss"], m["accuracy"])

    for rnd in range(1, num_rounds + 1):
        tr_loss, tr_acc = train(
            model, train_loader, epochs=epochs_per_round, lr=lr, device=device,
            momentum=momentum, weight_decay=weight_decay,
        )
        m = evaluate_detailed(model, test_loader, device)
        history.append({"round": rnd, **m, "train_loss": tr_loss, "train_accuracy": tr_acc})
        logger.info(
            "%s | round %2d | train loss=%.4f acc=%.4f | test loss=%.4f acc=%.4f macro_f1=%.4f",
            name, rnd, tr_loss, tr_acc, m["loss"], m["accuracy"], m["macro_f1"],
        )
    return history

"""Lightweight CNN plus the PyTorch train / test loops.

GroupNorm is used instead of BatchNorm on purpose: BatchNorm running statistics are buffers
that FedAvg does not average meaningfully and that break under non-IID data.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SmallCNN(nn.Module):
    """Four conv layers + two linear layers, ~0.2M parameters. Works for any input size >= 8."""

    def __init__(self, in_channels: int = 3, num_classes: int = 8, dropout: float = 0.3) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> list[nn.Module]:
            return [
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=cout),
                nn.ReLU(inplace=True),
            ]

        self.features = nn.Sequential(
            *block(in_channels, 32), *block(32, 32), nn.MaxPool2d(2),
            *block(32, 64), *block(64, 64), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(in_channels: int, num_classes: int, seed: int | None = None) -> SmallCNN:
    """Create a model. Passing a seed makes the initial weights identical across processes."""
    if seed is not None:
        torch.manual_seed(seed)
    return SmallCNN(in_channels=in_channels, num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------
# Weight (de)serialisation for Flower
# --------------------------------------------------------------------------------------
def get_weights(model: nn.Module) -> list[np.ndarray]:
    """Copy of every state_dict tensor as NumPy. Copies, so later training cannot mutate them."""
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    if len(keys) != len(weights):
        raise ValueError(f"Expected {len(keys)} weight arrays, received {len(weights)}.")
    state = OrderedDict((k, torch.as_tensor(np.asarray(w))) for k, w in zip(keys, weights))
    model.load_state_dict(state, strict=True)


# --------------------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------------------
def train(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    clip_norm: float = 5.0,
) -> tuple[float, float]:
    """Standard supervised loop. Returns (loss, accuracy) of the LAST epoch (training mode)."""
    if len(loader.dataset) == 0:
        raise ValueError("Cannot train on an empty dataset.")
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    epoch_loss, epoch_acc = 0.0, 0.0
    for epoch in range(epochs):
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became NaN/inf. Lower --lr.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
        epoch_loss, epoch_acc = total_loss / seen, correct / seen
        logger.debug("epoch %d/%d loss=%.4f acc=%.4f", epoch + 1, epochs, epoch_loss, epoch_acc)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_detailed(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Loss, accuracy and macro-F1. Macro-F1 matters because medical datasets are often imbalanced."""
    if len(loader.dataset) == 0:
        raise ValueError("Cannot evaluate on an empty dataset.")
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, seen = 0.0, 0
    preds_all, labels_all = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        seen += labels.size(0)
        preds_all.append(logits.argmax(dim=1).cpu())
        labels_all.append(labels.cpu())
    preds = torch.cat(preds_all).numpy()
    truth = torch.cat(labels_all).numpy()
    return {
        "loss": total_loss / seen,
        "accuracy": float((preds == truth).mean()),
        "macro_f1": float(f1_score(truth, preds, average="macro", zero_division=0)),
    }


def test(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Returns (loss, accuracy)."""
    m = evaluate_detailed(model, loader, device)
    return m["loss"], m["accuracy"]

"""Flower NumPyClient: one simulated hospital holding a private data shard."""

from __future__ import annotations

import logging
from typing import Callable

import torch
from flwr.client import NumPyClient
from flwr.common import Context
from flwr.common.logger import log
from torch.utils.data import DataLoader

from dataset import FederatedData, ImageTensorDataset
from model import build_model, evaluate_detailed, get_weights, set_weights, train
from utils import get_device

PARTITION_ID_KEY = "partition-id"  # key Flower's simulation engine uses in context.node_config


class FlowerClient(NumPyClient):
    """Trains locally on a private shard and validates on a private validation shard."""

    def __init__(
        self,
        cid: int,
        train_dataset: ImageTensorDataset,
        val_dataset: ImageTensorDataset,
        in_channels: int,
        num_classes: int,
        batch_size: int,
        device: torch.device,
        seed: int,
    ) -> None:
        super().__init__()
        self.cid = cid
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        # Weights are overwritten with the global parameters at the start of every fit/evaluate.
        self.model = build_model(in_channels, num_classes)

    # -- Flower API ---------------------------------------------------------------------
    def get_parameters(self, config):  # noqa: D401
        return get_weights(self.model)

    def fit(self, parameters, config):
        server_round = int(config.get("server_round", 0))
        epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.01))
        momentum = float(config.get("momentum", 0.9))
        weight_decay = float(config.get("weight_decay", 1e-4))

        set_weights(self.model, parameters)
        # Per-client, per-round seeding makes runs reproducible regardless of scheduling order.
        round_seed = self.seed + 1000 * self.cid + server_round
        torch.manual_seed(round_seed)
        generator = torch.Generator()
        generator.manual_seed(round_seed)
        loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )
        try:
            loss, acc = train(
                self.model, loader, epochs=epochs, lr=lr, device=self.device,
                momentum=momentum, weight_decay=weight_decay,
            )
        except Exception as exc:
            log(logging.ERROR, "Client %d failed during local training: %s", self.cid, exc)
            raise
        log(
            logging.INFO,
            "Client %d | round %d | fit  | n=%d epochs=%d | train loss=%.4f acc=%.4f",
            self.cid, server_round, len(self.train_dataset), epochs, loss, acc,
        )
        return get_weights(self.model), len(self.train_dataset), {
            "train_loss": float(loss),
            "train_accuracy": float(acc),
        }

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        try:
            m = evaluate_detailed(self.model, loader, self.device)
        except Exception as exc:
            log(logging.ERROR, "Client %d failed during validation: %s", self.cid, exc)
            raise
        log(
            logging.INFO,
            "Client %d | eval | n=%d | val loss=%.4f acc=%.4f",
            self.cid, len(self.val_dataset), m["loss"], m["accuracy"],
        )
        return float(m["loss"]), len(self.val_dataset), {
            "accuracy": float(m["accuracy"]),
            "macro_f1": float(m["macro_f1"]),
        }


def make_client_fn(
    data: FederatedData,
    batch_size: int,
    device: str,
    seed: int,
) -> Callable[[Context], object]:
    """Return the factory Flower calls to build a client for a given simulated node."""

    def client_fn(context: Context):
        torch.set_num_threads(1)  # avoid CPU oversubscription when several clients run at once
        cid = int(context.node_config[PARTITION_ID_KEY])
        if not 0 <= cid < len(data.client_train):
            raise IndexError(f"Partition id {cid} is outside [0, {len(data.client_train) - 1}].")
        client = FlowerClient(
            cid=cid,
            train_dataset=data.client_train[cid],
            val_dataset=data.client_val[cid],
            in_channels=data.num_channels,
            num_classes=data.num_classes,
            batch_size=batch_size,
            device=get_device(device),
            seed=seed,
        )
        return client.to_client()

    return client_fn

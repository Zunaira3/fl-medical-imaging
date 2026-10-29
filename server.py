"""Server side: FedAvg strategy plus a centralized-test-set evaluation hook."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from flwr.common import Metrics, NDArrays, Scalar, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from dataset import ImageTensorDataset
from model import build_model, evaluate_detailed, get_weights, set_weights

logger = logging.getLogger(__name__)


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Average each numeric client metric, weighted by the client's number of examples."""
    sums: dict[str, float] = {}
    weights: dict[str, int] = {}
    for num_examples, client_metrics in metrics:
        for key, value in client_metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            sums[key] = sums.get(key, 0.0) + num_examples * float(value)
            weights[key] = weights.get(key, 0) + num_examples
    return {k: sums[k] / weights[k] for k in sums if weights[k] > 0}


def get_on_fit_config_fn(local_epochs: int, lr: float, momentum: float, weight_decay: float):
    """Send hyper-parameters (and the round number) to clients before every fit."""

    def on_fit_config(server_round: int) -> dict[str, Scalar]:
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }

    return on_fit_config


def get_evaluate_fn(
    test_dataset: ImageTensorDataset,
    in_channels: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
    num_rounds: int,
    checkpoint_path: Optional[str] = None,
):
    """Build the server-side evaluation function.

    Flower calls it with the aggregated global weights at round 0 (initial model) and after
    every round, so the History object ends up holding the global convergence curve on the
    held-out centralized test set.
    """
    model = build_model(in_channels, num_classes)
    loader = DataLoader(test_dataset, batch_size=max(batch_size, 256), shuffle=False)

    def evaluate(server_round: int, parameters: NDArrays, config: dict[str, Scalar]):
        set_weights(model, parameters)
        m = evaluate_detailed(model, loader, device)
        logger.info(
            "SERVER | round %2d | global test loss=%.4f acc=%.4f macro_f1=%.4f",
            server_round, m["loss"], m["accuracy"], m["macro_f1"],
        )
        if checkpoint_path and server_round == num_rounds:
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            logger.info("SERVER | saved final global model to %s", path)
        return m["loss"], {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}

    return evaluate


def build_strategy(
    *,
    num_clients: int,
    num_rounds: int,
    test_dataset: ImageTensorDataset,
    in_channels: int,
    num_classes: int,
    batch_size: int,
    local_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    checkpoint_path: Optional[str] = None,
) -> FedAvg:
    """FedAvg over all clients every round, with a seeded shared initialisation."""
    init_model = build_model(in_channels, num_classes, seed=seed)
    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=get_evaluate_fn(
            test_dataset, in_channels, num_classes, batch_size, device, num_rounds, checkpoint_path
        ),
        on_fit_config_fn=get_on_fit_config_fn(local_epochs, lr, momentum, weight_decay),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters(get_weights(init_model)),
        accept_failures=False,
    )

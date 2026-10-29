"""Fast unit tests (no Ray, no downloads). Set RUN_SLOW=1 to also run the end-to-end simulation."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from client import FlowerClient
from config import ExperimentConfig
from dataset import (
    ImageTensorDataset,
    build_federated_data,
    make_synthetic,
    partition_like,
    partition_train,
)
from model import SmallCNN, build_model, count_parameters, get_weights, set_weights, train
from model import test as evaluate_model  # aliased so pytest does not collect it as a test
from server import weighted_average

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def bundle():
    return make_synthetic(n_train=600, n_val=120, n_test=240, seed=1)


def test_iid_partition_is_a_disjoint_cover():
    labels = np.random.default_rng(0).integers(0, 8, 1000)
    parts = partition_train(labels, 3, "iid", seed=0)
    merged = np.concatenate(parts)
    assert len(parts) == 3
    assert len(merged) == len(set(merged.tolist())) == 1000


def test_dirichlet_partition_is_a_disjoint_cover_and_skewed():
    labels = np.random.default_rng(0).integers(0, 8, 2000)
    parts = partition_train(labels, 3, "dirichlet", alpha=0.3, seed=0)
    merged = np.concatenate(parts)
    assert len(merged) == len(set(merged.tolist())) == 2000
    assert min(len(p) for p in parts) >= 10
    hists = [np.bincount(labels[p], minlength=8) / len(p) for p in parts]
    assert max(np.abs(hists[0] - hists[1])) > 0.05  # label distributions actually differ


def test_unknown_partition_scheme_raises():
    with pytest.raises(ValueError):
        partition_train(np.zeros(10, dtype=int), 2, "nope")


def test_validation_partition_is_nonempty_and_disjoint(bundle):
    fd = build_federated_data(bundle, 3, "dirichlet", 0.5, seed=3)
    assert all(len(v) > 0 for v in fd.client_val)
    assert sum(len(v) for v in fd.client_val) == len(bundle.val_y)
    train_parts = partition_train(bundle.train_y.numpy(), 3, "iid", seed=0)
    val_parts = partition_like(bundle.train_y.numpy(), train_parts, bundle.val_y.numpy(), seed=0)
    assert len(np.unique(np.concatenate(val_parts))) == len(bundle.val_y)


@pytest.mark.parametrize("channels,size", [(3, 28), (1, 28), (3, 64)])
def test_model_forward_shape(channels, size):
    model = SmallCNN(in_channels=channels, num_classes=5)
    out = model(torch.randn(4, channels, size, size))
    assert out.shape == (4, 5)
    assert count_parameters(model) < 1_000_000  # "lightweight"


def test_weight_roundtrip_is_exact():
    a, b = build_model(3, 8, seed=1), build_model(3, 8, seed=2)
    set_weights(b, get_weights(a))
    for wa, wb in zip(get_weights(a), get_weights(b)):
        assert np.array_equal(wa, wb)


def test_set_weights_rejects_wrong_length():
    model = build_model(3, 8, seed=0)
    with pytest.raises(ValueError):
        set_weights(model, get_weights(model)[:-1])


def test_same_seed_gives_same_initial_weights():
    a, b = build_model(3, 8, seed=7), build_model(3, 8, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(get_weights(a), get_weights(b)))


def test_training_improves_loss_and_accuracy(bundle):
    fd = build_federated_data(bundle, 3, "iid", 0.5, seed=0)
    model = build_model(fd.num_channels, fd.num_classes, seed=0)
    test_loader = DataLoader(fd.test, batch_size=128)
    loss0, acc0 = evaluate_model(model, test_loader, CPU)
    g = torch.Generator().manual_seed(0)
    loader = DataLoader(fd.full_train, batch_size=64, shuffle=True, generator=g)
    train(model, loader, epochs=4, lr=0.01, device=CPU)
    loss1, acc1 = evaluate_model(model, test_loader, CPU)
    assert loss1 < loss0
    assert acc1 > acc0


def test_train_rejects_empty_dataset():
    empty = ImageTensorDataset(torch.zeros(0, 3, 28, 28, dtype=torch.uint8), torch.zeros(0, dtype=torch.long))
    with pytest.raises(ValueError):
        train(build_model(3, 8, seed=0), DataLoader(empty, batch_size=4), 1, 0.01, CPU)


def test_weighted_average_math():
    out = weighted_average([(100, {"accuracy": 0.5}), (300, {"accuracy": 1.0}), (0, {"accuracy": 0.0})])
    assert out["accuracy"] == pytest.approx((100 * 0.5 + 300 * 1.0) / 400)


def test_fedavg_arithmetic_matches_manual_average():
    """FedAvg = example-weighted mean of client weights; verify the maths the strategy relies on."""
    w1, w2 = [np.ones((2, 2))], [np.full((2, 2), 3.0)]
    n1, n2 = 100, 300
    avg = [(n1 * a + n2 * b) / (n1 + n2) for a, b in zip(w1, w2)]
    assert np.allclose(avg[0], 2.5)


def test_client_fit_and_evaluate_contract(bundle):
    fd = build_federated_data(bundle, 3, "iid", 0.5, seed=0)
    client = FlowerClient(0, fd.client_train[0], fd.client_val[0], fd.num_channels, fd.num_classes, 32, CPU, 0)
    params = client.get_parameters({})
    new_params, n, metrics = client.fit(params, {"server_round": 1, "local_epochs": 1, "lr": 0.01})
    assert n == len(fd.client_train[0])
    assert len(new_params) == len(params)
    assert any(not np.array_equal(a, b) for a, b in zip(params, new_params))  # weights moved
    assert np.isfinite(metrics["train_loss"]) and 0.0 <= metrics["train_accuracy"] <= 1.0
    loss, n_val, ev = client.evaluate(new_params, {})
    assert n_val == len(fd.client_val[0]) and np.isfinite(loss) and 0.0 <= ev["accuracy"] <= 1.0


def test_config_validation():
    with pytest.raises(ValueError):
        ExperimentConfig(num_clients=1).validate()
    with pytest.raises(ValueError):
        ExperimentConfig(partition="bogus").validate()
    ExperimentConfig().validate()


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to run the Ray simulation")
def test_end_to_end_simulation(tmp_path):
    from run_simulation import main

    code = main([
        "--synthetic", "--rounds", "3", "--synthetic-train", "600", "--synthetic-val", "120",
        "--synthetic-test", "240", "--output-dir", str(tmp_path), "--partition", "dirichlet",
    ])
    assert code == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert set(res["histories"]) == {"Federated (FedAvg)", "Centralized", "Local-only (client 0)"}
    assert all(len(h) == 4 for h in res["histories"].values())
    assert (tmp_path / "convergence.png").exists() and (tmp_path / "global_model.pt").exists()

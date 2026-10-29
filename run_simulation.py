"""One-click orchestration: centralized baseline, federated simulation, comparison plot.

    python run_simulation.py                       # BloodMNIST, 3 clients, 20 rounds
    python run_simulation.py --synthetic --rounds 5   # offline smoke test (no download)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import flwr as fl
import torch
from flwr.server import History

from centralized import train_baseline
from client import make_client_fn
from config import ExperimentConfig
from dataset import DatasetBundle, build_federated_data, load_medmnist, make_synthetic
from plotting import plot_results
from server import build_strategy
from utils import get_device, save_json, set_seed, setup_logging

logger = logging.getLogger("run_simulation")


# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    d = ExperimentConfig()
    p = argparse.ArgumentParser(
        description="Federated (FedAvg) vs centralized training on a medical image dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default=d.dataset, help="Single-label 2D MedMNIST flag.")
    p.add_argument("--data-dir", default=d.data_dir)
    p.add_argument("--image-size", type=int, default=d.image_size, choices=[28, 64, 128, 224])
    p.add_argument("--synthetic", action="store_true", help="Use generated data (offline smoke test only).")
    p.add_argument("--num-clients", type=int, default=d.num_clients)
    p.add_argument("--rounds", type=int, default=d.num_rounds, dest="num_rounds")
    p.add_argument("--local-epochs", type=int, default=d.local_epochs)
    p.add_argument("--partition", choices=["iid", "dirichlet"], default=d.partition)
    p.add_argument("--dirichlet-alpha", type=float, default=d.dirichlet_alpha)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--momentum", type=float, default=d.momentum)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device, choices=["auto", "cpu", "cuda"])
    p.add_argument("--ray-num-cpus", type=int, default=d.ray_num_cpus)
    p.add_argument("--output-dir", default=d.output_dir)
    p.add_argument("--skip-local-baseline", action="store_true")
    p.add_argument("--tolerance", type=float, default=d.tolerance,
                   help="Max accuracy gap (absolute) to call federated 'comparable' to centralized.")
    p.add_argument("--synthetic-train", type=int, default=d.synthetic_train)
    p.add_argument("--synthetic-val", type=int, default=d.synthetic_val)
    p.add_argument("--synthetic-test", type=int, default=d.synthetic_test)
    ns = p.parse_args(argv)
    cfg = ExperimentConfig(**{k: v for k, v in vars(ns).items()})
    cfg.validate()
    return cfg


def load_bundle(cfg: ExperimentConfig) -> DatasetBundle:
    if cfg.synthetic:
        return make_synthetic(
            n_train=cfg.synthetic_train, n_val=cfg.synthetic_val, n_test=cfg.synthetic_test,
            size=cfg.image_size, seed=cfg.seed,
        )
    return load_medmnist(cfg.dataset, cfg.data_dir, size=cfg.image_size)


def history_to_records(history: History, num_rounds: int) -> list[dict[str, float]]:
    """Convert Flower's History into one flat record per round (round 0 = initial model)."""
    losses = dict(history.losses_centralized)
    if not losses:
        raise RuntimeError("Flower returned no centralized metrics; the simulation did not run.")
    acc = dict(history.metrics_centralized.get("accuracy", []))
    f1 = dict(history.metrics_centralized.get("macro_f1", []))
    tr_loss = dict(history.metrics_distributed_fit.get("train_loss", []))
    tr_acc = dict(history.metrics_distributed_fit.get("train_accuracy", []))
    val_loss = dict(history.losses_distributed)
    val_acc = dict(history.metrics_distributed.get("accuracy", []))
    nan = float("nan")
    records = []
    for rnd in sorted(losses):
        records.append({
            "round": int(rnd),
            "loss": float(losses[rnd]),
            "accuracy": float(acc.get(rnd, nan)),
            "macro_f1": float(f1.get(rnd, nan)),
            "train_loss": float(tr_loss.get(rnd, nan)),
            "train_accuracy": float(tr_acc.get(rnd, nan)),
            "client_val_loss": float(val_loss.get(rnd, nan)),
            "client_val_accuracy": float(val_acc.get(rnd, nan)),
        })
    if len(records) != num_rounds + 1:
        raise RuntimeError(
            f"Expected {num_rounds + 1} evaluation points but got {len(records)}. "
            "The simulation ended early; check the log for client errors."
        )
    return records


def _tail_mean(records: list[dict[str, float]], key: str) -> float:
    """Mean over the last few rounds (up to 3, fewer for short runs) to damp round-to-round noise."""
    k = max(1, min(3, (len(records) - 1) // 5))
    vals = [r[key] for r in records[-k:]]
    return sum(vals) / len(vals)


def summarize(histories: dict[str, list[dict[str, float]]], tolerance: float) -> dict:
    """Final-round and tail-mean metrics, plus the federated-vs-centralized gap."""
    summary: dict = {"methods": {}}
    for name, h in histories.items():
        summary["methods"][name] = {
            "final_accuracy": h[-1]["accuracy"],
            "final_macro_f1": h[-1]["macro_f1"],
            "final_loss": h[-1]["loss"],
            "tail_mean_accuracy": _tail_mean(h, "accuracy"),
            "best_accuracy": max(r["accuracy"] for r in h),
        }
    fed, cen = summary["methods"]["Federated (FedAvg)"], summary["methods"]["Centralized"]
    gap = cen["tail_mean_accuracy"] - fed["tail_mean_accuracy"]
    summary["gap_centralized_minus_federated"] = gap
    summary["tolerance"] = tolerance
    summary["comparable"] = bool(abs(gap) <= tolerance)
    return summary


def log_summary(summary: dict) -> None:
    logger.info("=" * 78)
    logger.info("%-26s %10s %10s %12s %10s", "Method", "final acc", "final F1", "tail acc", "best acc")
    for name, m in summary["methods"].items():
        logger.info(
            "%-26s %10.4f %10.4f %12.4f %10.4f",
            name, m["final_accuracy"], m["final_macro_f1"], m["tail_mean_accuracy"], m["best_accuracy"],
        )
    gap = summary["gap_centralized_minus_federated"]
    verdict = "COMPARABLE" if summary["comparable"] else "NOT comparable"
    logger.info("Gap (centralized - federated, tail-mean accuracy): %+.4f | tolerance %.3f -> %s",
                gap, summary["tolerance"], verdict)
    logger.info("Single seed, single run: treat this as a demonstration, not a statistical result.")
    logger.info("=" * 78)


# --------------------------------------------------------------------------------------
def run(cfg: ExperimentConfig) -> dict:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger.info("Device: %s | flwr %s | torch %s", device, fl.__version__, torch.__version__)

    bundle = load_bundle(cfg)
    data = build_federated_data(bundle, cfg.num_clients, cfg.partition, cfg.dirichlet_alpha, cfg.seed)
    common = dict(
        in_channels=data.num_channels, num_classes=data.num_classes,
        num_rounds=cfg.num_rounds, epochs_per_round=cfg.local_epochs,
        batch_size=cfg.batch_size, lr=cfg.lr, momentum=cfg.momentum,
        weight_decay=cfg.weight_decay, device=device, seed=cfg.seed,
    )

    histories: dict[str, list[dict[str, float]]] = {}

    # 1) Centralized baseline: all data pooled in one place.
    t0 = time.time()
    logger.info("--- Centralized baseline (pooled training set) ---")
    histories["Centralized"] = train_baseline("CENTRAL", data.full_train, data.test, **common)
    logger.info("Centralized baseline finished in %.1fs", time.time() - t0)

    # 2) Optional: one client training alone on its own shard (what you get WITHOUT federation).
    if not cfg.skip_local_baseline:
        t0 = time.time()
        logger.info("--- Local-only baseline (client 0 shard) ---")
        histories["Local-only (client 0)"] = train_baseline("LOCAL0", data.client_train[0], data.test, **common)
        logger.info("Local-only baseline finished in %.1fs", time.time() - t0)

    # 3) Federated simulation with FedAvg.
    t0 = time.time()
    logger.info("--- Federated simulation: %d clients, %d rounds, FedAvg ---", cfg.num_clients, cfg.num_rounds)
    strategy = build_strategy(
        num_clients=cfg.num_clients, num_rounds=cfg.num_rounds, test_dataset=data.test,
        in_channels=data.num_channels, num_classes=data.num_classes, batch_size=cfg.batch_size,
        local_epochs=cfg.local_epochs, lr=cfg.lr, momentum=cfg.momentum,
        weight_decay=cfg.weight_decay, device=device, seed=cfg.seed,
        checkpoint_path=str(out / "global_model.pt"),
    )
    client_device = device.type
    num_gpus = (1.0 / cfg.num_clients) if client_device == "cuda" else 0.0
    ray_cpus = cfg.ray_num_cpus or min(cfg.num_clients, os.cpu_count() or 1)
    try:
        fl_history = fl.simulation.start_simulation(
            client_fn=make_client_fn(data, cfg.batch_size, client_device, cfg.seed),
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": num_gpus},
            ray_init_args={"num_cpus": ray_cpus, "include_dashboard": False, "ignore_reinit_error": True},
        )
    except Exception:
        logger.exception("Federated simulation failed.")
        raise
    histories["Federated (FedAvg)"] = history_to_records(fl_history, cfg.num_rounds)
    logger.info("Federated simulation finished in %.1fs", time.time() - t0)

    # 4) Persist, plot, summarise.
    summary = summarize(histories, cfg.tolerance)
    save_json({"config": cfg.to_dict(), "histories": histories, "summary": summary}, out / "results.json")
    title = (f"{data.dataset_name} | {cfg.num_clients} clients | {cfg.partition}"
             + (f" (alpha={cfg.dirichlet_alpha})" if cfg.partition == "dirichlet" else "")
             + (" | SYNTHETIC DATA" if cfg.synthetic else ""))
    plot_path = plot_results(histories, out / "convergence.png", title)
    log_summary(summary)
    logger.info("Saved: %s, %s, %s", out / "results.json", plot_path, out / "global_model.pt")
    return summary


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    setup_logging(Path(cfg.output_dir) / "run.log")
    logger.info("Config: %s", cfg.to_dict())
    try:
        run(cfg)
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level guard, full traceback already logged
        logger.error("Run failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

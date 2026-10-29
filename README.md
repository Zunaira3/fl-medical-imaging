# Federated Medical Image Classification (PyTorch + Flower)

Simulates a privacy-motivated federated setup: **3 client nodes** (hospitals) and **1 aggregating server**
train a small CNN on **BloodMNIST** (8-class blood-cell microscopy, MedMNIST) using **FedAvg**.
The run compares the federated model's global test accuracy against a **centralized baseline**
(and a **local-only baseline**: one client training alone) and writes a convergence plot.

> **What "privacy-preserving" means here.** Raw images never leave a client; only model weights do.
> This project does **not** implement differential privacy or secure aggregation, and weight updates
> can still leak information. Do not describe it as formally private.

## Repository layout

```
fl-medical-imaging/
├── README.md
├── requirements.txt      pinned dependencies (tested versions)
├── pytest.ini
├── config.py             ExperimentConfig dataclass (all hyper-parameters)
├── utils.py              logging, seeding, device selection, JSON helpers
├── dataset.py            MedMNIST loader, synthetic fallback, IID / Dirichlet partitioning
├── model.py              SmallCNN, train(), test(), evaluate_detailed(), weight (de)serialisation
├── client.py             FlowerClient (NumPyClient) + client factory for the simulator
├── server.py             FedAvg strategy, server-side centralized-test evaluation, checkpointing
├── centralized.py        centralized and local-only baselines (same model, same optimiser)
├── plotting.py           3-panel convergence figure
├── run_simulation.py     one-command entry point
└── tests/
    └── test_pipeline.py  unit tests + optional end-to-end test
```

## Setup

Python 3.12 was used for testing.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`flwr[simulation]` installs Ray, which powers the simulated clients. The PyPI `torch` wheel on Linux
pulls CUDA libraries (several GB). For a CPU-only install, use PyTorch's CPU index first, see
https://pytorch.org/get-started/locally/.

## Run

```bash
# Real data (downloads BloodMNIST, ~35 MB, on first run), IID split, 20 rounds
python run_simulation.py

# Harder, more realistic: label-skewed (non-IID) clients
python run_simulation.py --partition dirichlet --dirichlet-alpha 0.5

# Another MedMNIST dataset (binary chest X-ray pneumonia)
python run_simulation.py --dataset pneumoniamnist

# Offline smoke test with generated data (NOT medical data, only checks the plumbing)
python run_simulation.py --synthetic --rounds 5
```

Useful flags (`python run_simulation.py --help` lists all): `--rounds`, `--local-epochs`, `--lr`,
`--batch-size`, `--num-clients`, `--seed`, `--device {auto,cpu,cuda}`, `--ray-num-cpus`,
`--image-size {28,64,128,224}`, `--skip-local-baseline`, `--tolerance`, `--output-dir`.

### Outputs (in `outputs/`)

| File | Content |
|---|---|
| `convergence.png` | Test accuracy, test loss and macro-F1 vs communication round for all methods |
| `results.json` | Config, per-round metrics for every method, final summary |
| `global_model.pt` | Final aggregated global model (`state_dict`) |
| `run.log` | Server, baseline and driver logs. Per-client logs are printed to the console by Ray |

## Tests

```bash
pytest                    # fast unit tests, no Ray, no download
RUN_SLOW=1 pytest         # also runs the full Ray simulation on synthetic data
```

## How it works

* **Data**: official MedMNIST train split is partitioned across clients (`iid` or Dirichlet label skew).
  Each client's validation shard mirrors its training class mix. The official **test split stays on the
  server** and is used only to measure the global model.
* **Model**: 4-conv CNN with GroupNorm (about 0.2M parameters). GroupNorm replaces BatchNorm because
  BatchNorm running statistics are not meaningfully averaged by FedAvg.
* **Federation**: `FedAvg`, all 3 clients every round, `local_epochs` epochs of SGD with momentum per round.
  The server evaluates the aggregated weights each round (`evaluate_fn`), including round 0.
* **Fair comparison**: all methods start from identical seeded initial weights and use the same optimiser.
  One communication round equals `local_epochs` epochs over the pooled data, so the centralized baseline
  gets the same number of passes per "round".
* **Metrics**: loss, accuracy and macro-F1 (macro-F1 matters on imbalanced medical data).
* **Verdict**: "comparable" means the mean accuracy over the last few rounds differs by at most
  `--tolerance` (default 0.03 absolute). That threshold is a convention, not a statistical test.

## Known limitations

* `flwr.simulation.start_simulation` is **deprecated** (since Flower 1.13) in favour of `flwr run` with
  `ServerApp` / `ClientApp`. It works in the pinned `flwr==1.42.0`, so keep the pin. Upgrading Flower
  may require porting `run_simulation.py`.
* Single seed per run. Run several `--seed` values before drawing conclusions.
* Non-IID partitions typically widen the gap to the centralized baseline, which is expected FedAvg behaviour.
* GPU mode (`--device cuda`, fractional GPU per client) is implemented but was not exercised in testing.
* No standalone multi-machine deployment (SuperLink/SuperNode), no differential privacy, no secure aggregation.

## Troubleshooting

* **Offline / download fails**: place `bloodmnist.npz` in `data/` manually, or use `--synthetic`.
* **Ray memory or disk warnings**: lower `--ray-num-cpus`, or free disk space (Ray writes to `/tmp/ray`).
* **Loss is NaN**: lower `--lr`.

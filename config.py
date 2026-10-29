"""Experiment configuration shared by every entry point."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    # Data
    dataset: str = "bloodmnist"      # any 2D single-label MedMNIST flag
    data_dir: str = "data"
    image_size: int = 28             # 28, 64, 128 or 224 (larger = much slower)
    synthetic: bool = False          # offline smoke-test data, NOT a medical dataset
    # Federation
    num_clients: int = 3
    num_rounds: int = 20
    local_epochs: int = 1
    partition: str = "iid"           # "iid" or "dirichlet"
    dirichlet_alpha: float = 0.5
    # Optimisation (identical for federated and centralized runs)
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    # Run control
    seed: int = 42
    device: str = "auto"
    ray_num_cpus: int | None = None  # None = min(num_clients, os.cpu_count())
    output_dir: str = "outputs"
    skip_local_baseline: bool = False
    tolerance: float = 0.03          # max |federated - centralized| accuracy gap called "comparable"
    # Synthetic-data sizing (only used when synthetic=True)
    synthetic_train: int = 1800
    synthetic_val: int = 300
    synthetic_test: int = 600
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.num_clients < 2:
            raise ValueError("num_clients must be >= 2.")
        if self.num_rounds < 1 or self.local_epochs < 1:
            raise ValueError("num_rounds and local_epochs must be >= 1.")
        if self.partition not in {"iid", "dirichlet"}:
            raise ValueError("partition must be 'iid' or 'dirichlet'.")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0.")
        if self.batch_size < 1 or self.lr <= 0:
            raise ValueError("batch_size must be >= 1 and lr > 0.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

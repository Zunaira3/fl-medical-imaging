"""Dataset loading and federated partitioning.

Default dataset: BloodMNIST (MedMNIST v2/v3), 8-class blood-cell microscopy, 28x28 RGB.
Any single-label 2D MedMNIST flag works through --dataset (e.g. pneumoniamnist, pathmnist).

Images are kept as uint8 tensors and normalised on the fly, which keeps memory small and
keeps the objects that Ray has to serialise for each simulated client light.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Normalisation constants: maps uint8 [0, 255] to roughly [-1, 1].
_MEAN = 0.5
_STD = 0.5


@dataclass
class DatasetBundle:
    """Raw tensors for the official train / val / test splits."""

    name: str
    num_classes: int
    num_channels: int
    train_x: torch.Tensor  # uint8, N x C x H x W
    train_y: torch.Tensor  # int64, N
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


class ImageTensorDataset(Dataset):
    """Wraps uint8 NCHW images and int64 labels; normalises per item."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if images.shape[0] != labels.shape[0]:
            raise ValueError("images and labels must have the same length.")
        if images.dtype != torch.uint8:
            raise TypeError("images must be a uint8 tensor.")
        self.images = images
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx].float().div_(255.0).sub_(_MEAN).div_(_STD)
        return x, self.labels[idx]

    def subset(self, indices: np.ndarray) -> "ImageTensorDataset":
        idx = torch.as_tensor(np.asarray(indices, dtype=np.int64))
        return ImageTensorDataset(self.images[idx], self.labels[idx])


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _to_nchw(imgs: np.ndarray) -> torch.Tensor:
    """MedMNIST stores N x H x W (grayscale) or N x H x W x C (colour) uint8 arrays."""
    arr = np.asarray(imgs)
    if arr.ndim == 3:
        arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"Unexpected image array shape {arr.shape}.")
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2))).to(torch.uint8)


def load_medmnist(name: str, root: str | Path, size: int = 28) -> DatasetBundle:
    """Load a single-label 2D MedMNIST dataset, downloading it the first time."""
    try:
        import medmnist
        from medmnist import INFO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("medmnist is not installed. Run: pip install -r requirements.txt") from exc

    name = name.lower()
    if name not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset '{name}'. Options: {sorted(INFO)}")
    info = INFO[name]
    if "multi-label" in info["task"]:
        raise ValueError(f"'{name}' is multi-label; this project supports single-label classification only.")
    if "n_channels" not in info or not hasattr(medmnist, info["python_class"]):
        raise ValueError(f"'{name}' is not a supported 2D MedMNIST dataset.")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cls = getattr(medmnist, info["python_class"])
    suffix = "" if size == 28 else f"_{size}"
    npz_path = root / f"{name}{suffix}.npz"
    need_download = not npz_path.exists()
    if need_download:
        logger.info("Downloading %s (%dx%d) to %s ...", name, size, size, root)

    splits = {}
    for split in ("train", "val", "test"):
        try:
            ds = cls(split=split, download=need_download, root=str(root), size=size)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load MedMNIST '{name}'. If you are offline, download "
                f"'{npz_path.name}' manually into '{root}' or run with --synthetic."
            ) from exc
        splits[split] = (_to_nchw(ds.imgs), torch.from_numpy(np.asarray(ds.labels).reshape(-1)).long())
        need_download = False  # the file now exists

    n_classes = len(info["label"])
    bundle = DatasetBundle(
        name=name,
        num_classes=n_classes,
        num_channels=int(info["n_channels"]),
        train_x=splits["train"][0], train_y=splits["train"][1],
        val_x=splits["val"][0], val_y=splits["val"][1],
        test_x=splits["test"][0], test_y=splits["test"][1],
    )
    _validate_bundle(bundle)
    logger.info(
        "Loaded %s: train=%d val=%d test=%d classes=%d channels=%d",
        name, len(bundle.train_y), len(bundle.val_y), len(bundle.test_y),
        bundle.num_classes, bundle.num_channels,
    )
    return bundle


def make_synthetic(
    n_train: int = 1800,
    n_val: int = 300,
    n_test: int = 600,
    num_classes: int = 8,
    num_channels: int = 3,
    size: int = 28,
    seed: int = 0,
) -> DatasetBundle:
    """Class-conditional synthetic images for offline smoke tests.

    This is NOT medical data. It only exercises the pipeline end to end.
    """
    rng = np.random.default_rng(seed)
    coarse = 7
    templates = rng.random((num_classes, num_channels, coarse, coarse))
    reps = int(np.ceil(size / coarse))
    templates = np.kron(templates, np.ones((1, 1, reps, reps)))[:, :, :size, :size]

    def _make(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = np.arange(n) % num_classes
        rng.shuffle(labels)
        noise = rng.normal(0.0, 0.35, size=(n, num_channels, size, size))
        gain = rng.uniform(0.8, 1.2, size=(n, 1, 1, 1))
        imgs = np.clip(templates[labels] * gain + noise, 0.0, 1.0)
        return torch.from_numpy((imgs * 255).astype(np.uint8)), torch.from_numpy(labels.astype(np.int64))

    tr, va, te = _make(n_train), _make(n_val), _make(n_test)
    bundle = DatasetBundle("synthetic", num_classes, num_channels, tr[0], tr[1], va[0], va[1], te[0], te[1])
    _validate_bundle(bundle)
    logger.warning("Using SYNTHETIC data (train=%d val=%d test=%d). Not a medical dataset.", n_train, n_val, n_test)
    return bundle


def _validate_bundle(b: DatasetBundle) -> None:
    for split, y in (("train", b.train_y), ("val", b.val_y), ("test", b.test_y)):
        if y.numel() == 0:
            raise ValueError(f"The {split} split is empty.")
        if int(y.min()) < 0 or int(y.max()) >= b.num_classes:
            raise ValueError(f"{split} labels fall outside [0, {b.num_classes - 1}].")


# --------------------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------------------
def partition_train(
    labels: np.ndarray,
    num_clients: int,
    scheme: str = "iid",
    alpha: float = 0.5,
    seed: int = 0,
    min_size: int = 10,
) -> list[np.ndarray]:
    """Split sample indices across clients, either IID or Dirichlet label-skewed (non-IID)."""
    labels = np.asarray(labels)
    n = len(labels)
    rng = np.random.default_rng(seed)

    if scheme == "iid":
        perm = rng.permutation(n)
        parts = [np.sort(p) for p in np.array_split(perm, num_clients)]
    elif scheme == "dirichlet":
        classes = np.unique(labels)
        parts = None
        for _attempt in range(200):
            buckets: list[list[int]] = [[] for _ in range(num_clients)]
            for c in classes:
                idx = np.where(labels == c)[0]
                rng.shuffle(idx)
                props = rng.dirichlet(np.full(num_clients, alpha))
                cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
                for k, chunk in enumerate(np.split(idx, cuts)):
                    buckets[k].extend(chunk.tolist())
            if min(len(b) for b in buckets) >= min_size:
                parts = [np.sort(np.array(b, dtype=np.int64)) for b in buckets]
                break
        if parts is None:
            raise RuntimeError(
                f"Could not build a Dirichlet partition with >= {min_size} samples per client. "
                "Increase --dirichlet-alpha or use --partition iid."
            )
    else:
        raise ValueError(f"Unknown partition scheme '{scheme}'.")

    if min(len(p) for p in parts) < 1:
        raise RuntimeError("A client received zero training samples.")
    return parts


def partition_like(
    train_labels: np.ndarray,
    train_parts: list[np.ndarray],
    other_labels: np.ndarray,
    seed: int = 0,
) -> list[np.ndarray]:
    """Split another set (e.g. validation) so each client's class mix mirrors its training shard."""
    rng = np.random.default_rng(seed)
    train_labels = np.asarray(train_labels)
    other_labels = np.asarray(other_labels)
    k = len(train_parts)
    buckets: list[list[int]] = [[] for _ in range(k)]
    for c in np.unique(train_labels):
        counts = np.array([(train_labels[p] == c).sum() for p in train_parts], dtype=float)
        if counts.sum() == 0:
            continue
        props = counts / counts.sum()
        idx = np.where(other_labels == c)[0]
        rng.shuffle(idx)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for i, chunk in enumerate(np.split(idx, cuts)):
            buckets[i].extend(chunk.tolist())
    parts = [np.sort(np.array(b, dtype=np.int64)) for b in buckets]
    if min(len(p) for p in parts) < 1:
        raise RuntimeError("A client received zero validation samples; the validation split is too small.")
    return parts


def class_histogram(labels: np.ndarray, num_classes: int) -> list[int]:
    return np.bincount(np.asarray(labels), minlength=num_classes).tolist()


@dataclass
class FederatedData:
    """Everything the simulation needs: per-client shards plus the central test set."""

    client_train: list[ImageTensorDataset]
    client_val: list[ImageTensorDataset]
    full_train: ImageTensorDataset
    full_val: ImageTensorDataset
    test: ImageTensorDataset
    num_classes: int
    num_channels: int
    dataset_name: str


def build_federated_data(
    bundle: DatasetBundle,
    num_clients: int,
    scheme: str,
    alpha: float,
    seed: int,
) -> FederatedData:
    """Partition the official train/val splits across clients; keep the test split central."""
    train_y = bundle.train_y.numpy()
    train_parts = partition_train(train_y, num_clients, scheme, alpha, seed)
    val_parts = partition_like(train_y, train_parts, bundle.val_y.numpy(), seed)

    full_train = ImageTensorDataset(bundle.train_x, bundle.train_y)
    full_val = ImageTensorDataset(bundle.val_x, bundle.val_y)
    test = ImageTensorDataset(bundle.test_x, bundle.test_y)

    client_train = [full_train.subset(p) for p in train_parts]
    client_val = [full_val.subset(p) for p in val_parts]

    for i, (tp, vp) in enumerate(zip(train_parts, val_parts)):
        logger.info(
            "Client %d | train=%d val=%d | train class counts=%s",
            i, len(tp), len(vp), class_histogram(train_y[tp], bundle.num_classes),
        )
    return FederatedData(
        client_train=client_train,
        client_val=client_val,
        full_train=full_train,
        full_val=full_val,
        test=test,
        num_classes=bundle.num_classes,
        num_channels=bundle.num_channels,
        dataset_name=bundle.name,
    )

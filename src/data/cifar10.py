from collections.abc import Sequence
import time
import warnings
from urllib.error import HTTPError, URLError

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DEFAULT_CIFAR10_MIRRORS = [
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
    "https://zenodo.org/records/10089977/files/cifar-10-python.tar.gz?download=1",
]


def _class_limited_indices(targets: Sequence[int], class_counts: Sequence[int], seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    indices: list[int] = []
    for cls, count in enumerate(class_counts):
        cls_indices = np.flatnonzero(targets_array == cls)
        if count > len(cls_indices):
            raise ValueError(f"class {cls} requested {count} samples but only {len(cls_indices)} exist")
        chosen = rng.choice(cls_indices, size=count, replace=False)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return indices


def _load_cifar10(
    root: str,
    train: bool,
    download: bool,
    transform: transforms.Compose,
    mirrors: Sequence[str],
    download_retries: int,
    retry_sleep_seconds: float,
) -> datasets.CIFAR10:
    urls = [datasets.CIFAR10.url, *mirrors]
    last_error: Exception | None = None
    for url in urls:
        dataset_cls = type("MirrorCIFAR10", (datasets.CIFAR10,), {"url": url})
        attempts = max(download_retries, 1) if download else 1
        for attempt in range(1, attempts + 1):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                        category=Warning,
                        module=r"torchvision\.datasets\.cifar",
                    )
                    return dataset_cls(root=root, train=train, download=download, transform=transform)
            except (HTTPError, URLError, RuntimeError) as exc:
                last_error = exc
                if not download or attempt == attempts:
                    break
                time.sleep(retry_sleep_seconds)
    raise RuntimeError(
        "Could not load CIFAR-10. The default torchvision URL may be unavailable. "
        "Either retry later, set data.download=false after placing the extracted "
        "`cifar-10-batches-py` directory under data.root, or add a working URL to "
        "data.mirrors in the YAML config."
    ) from last_error


def build_cifar10_train_loader(config: dict, seed: int) -> tuple[DataLoader, dict]:
    data_cfg = config.get("data", {})
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = _load_cifar10(
        root=data_cfg.get("root", "data"),
        train=True,
        download=bool(data_cfg.get("download", False)),
        transform=transform,
        mirrors=data_cfg.get("mirrors", DEFAULT_CIFAR10_MIRRORS),
        download_retries=int(data_cfg.get("download_retries", 3)),
        retry_sleep_seconds=float(data_cfg.get("retry_sleep_seconds", 5.0)),
    )
    class_counts = data_cfg.get("class_counts", "balanced")
    if class_counts == "balanced":
        effective_counts = [5000] * 10
        train_dataset = dataset
    else:
        effective_counts = [int(v) for v in class_counts]
        train_dataset = Subset(dataset, _class_limited_indices(dataset.targets, effective_counts, seed))

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        generator=generator,
        persistent_workers=int(data_cfg.get("num_workers", 4)) > 0,
    )
    info = {
        "class_counts": effective_counts,
        "num_samples": len(train_dataset),
        "batch_size": int(data_cfg.get("batch_size", 256)),
    }
    return loader, info

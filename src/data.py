from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class LensDataset(Dataset):
    """
    Dataset for gravitational lens binary classification.

    Supports loading from NumPy files (.npy/.npz) and image formats.
    Output tensors: (C, H, W) float32
    """

    def __init__(self, samples: Sequence[Sample], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]

        # Load image
        image = self._load_file(sample.path)

        # Convert to tensor
        tensor = torch.from_numpy(image)

        # Apply transforms
        if self.transform is not None:
            tensor = self.transform(tensor)

        # DOMAIN ADAPTATION (Noise Injection)
        tensor = tensor + torch.randn_like(tensor) * 0.001
        tensor = torch.clamp(tensor, 0.0, 1.0)

        return tensor, sample.label

    @staticmethod
    def _load_file(path: Path) -> np.ndarray:
        suffix = path.suffix.lower()

        if suffix == ".npy":
            arr = np.load(path)

        elif suffix == ".npz":
            data = np.load(path)
            first_key = data.files[0]
            arr = data[first_key]

        else:
            arr = np.asarray(Image.open(path).convert("RGB"))

        arr = _to_chw_float32(arr)
        return arr


def _to_chw_float32(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    elif arr.ndim == 3:
        if arr.shape[0] == 3:
            # already CHW
            pass
        elif arr.shape[-1] == 3:
            # convert HWC → CHW
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported shape: {arr.shape}")

    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    arr = arr.astype(np.float32)

    if arr.max() > 1.0:
        arr = arr / 255.0

    return arr


def collect_samples(class_dir: Path, label: int) -> List[Sample]:
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing directory: {class_dir}")

    samples: List[Sample] = []

    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            samples.append(Sample(path=path, label=label))

    if not samples:
        raise ValueError(f"No files found in: {class_dir}")

    return samples


def build_train_test_samples(data_root: Path) -> Tuple[List[Sample], List[Sample]]:
    train_lens = collect_samples(data_root / "train_lenses", label=1)
    train_nonlens = collect_samples(data_root / "train_nonlenses", label=0)

    test_lens = collect_samples(data_root / "test_lenses", label=1)
    test_nonlens = collect_samples(data_root / "test_nonlenses", label=0)

    train_samples = train_lens + train_nonlens
    test_samples = test_lens + test_nonlens

    return train_samples, test_samples
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# Standard CIFAR-10 normalization constants (mean/std per RGB channel)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


@dataclass
class DataConfig:
    # Folder containing data_batch_1..5 and test_batch
    data_dir: str
    batch_size: int = 128
    num_workers: int = 2
    image_size: int = 32
    augment: bool = True


def _unpickle(file_path: Path) -> dict:
    # CIFAR-10 batch files are pickled Python dicts
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")
    return data_dict


def _load_cifar10_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads CIFAR-10 from the python batch format.

    Returns:
      X_train: (50000, 32, 32, 3) uint8
      y_train: (50000,) int64
      X_test:  (10000, 32, 32, 3) uint8
      y_test:  (10000,) int64
    """
    data_dir = Path(data_dir)

    train_files = [data_dir / f"data_batch_{i}" for i in range(1, 6)]
    test_file = data_dir / "test_batch"

    train_images_list = []
    train_labels_list = []

    # Load the 5 training batches (each has 10,000 images)
    for fp in train_files:
        batch = _unpickle(fp)

        # batch[b"data"] shape: (10000, 3072)
        # 3072 = 3 * 32 * 32 (R then G then B)
        data = batch[b"data"]
        labels = batch[b"labels"]

        train_images_list.append(data)
        train_labels_list.append(labels)

    # Concatenate -> 50,000 training images
    X_train = np.concatenate(train_images_list, axis=0)  # (50000, 3072)
    y_train = np.concatenate(train_labels_list, axis=0)  # (50000,)

    # Load test batch -> 10,000 images
    test_batch = _unpickle(test_file)
    X_test = test_batch[b"data"]               # (10000, 3072)
    y_test = np.array(test_batch[b"labels"])   # (10000,)

    # Reshape flat vectors -> (N, 3, 32, 32)
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    # Convert to (N, 32, 32, 3) so PIL can read it
    X_train = X_train.transpose(0, 2, 3, 1).astype(np.uint8)
    X_test = X_test.transpose(0, 2, 3, 1).astype(np.uint8)

    # Labels should be int64 for PyTorch classification losses
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    return X_train, y_train, X_test, y_test


def build_transforms(cfg: DataConfig) -> Tuple[T.Compose, T.Compose]:
    # Train transforms often include augmentation; test transforms should not.
    if cfg.augment:
        train_tfms = T.Compose([
            T.RandomCrop(cfg.image_size, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),  # PIL (H,W,C) -> Tensor (C,H,W) float in [0,1]
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_tfms = T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    test_tfms = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_tfms, test_tfms


class CIFAR10Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images          # (N, 32, 32, 3) uint8
        self.labels = labels          # (N,) int64
        self.transform = transform
        assert len(self.images) == len(self.labels), "Images/labels length mismatch!"

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]    # (32, 32, 3) uint8
        label = self.labels[idx]  # int

        # Convert numpy image -> PIL Image for torchvision transforms
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)  # becomes Tensor (3,32,32), normalized

        label = torch.tensor(label, dtype=torch.long)
        return img, label  # tuple: (image_tensor, label_tensor)


def get_cifar10_loaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    X_train, y_train, X_test, y_test = _load_cifar10_data(cfg.data_dir)
    train_tfms, test_tfms = build_transforms(cfg)

    train_ds = CIFAR10Dataset(X_train, y_train, transform=train_tfms)
    test_ds = CIFAR10Dataset(X_test, y_test, transform=test_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader

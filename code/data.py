import os
from typing import Dict, Iterator, List, Tuple

import numpy as np
from PIL import Image


def load_eurosat(data_root: str, image_size: Tuple[int, int] = (64, 64)) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    class_names = sorted(
        [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    )
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    xs, ys = [], []
    for class_name in class_names:
        class_dir = os.path.join(data_root, class_name)
        for fname in sorted(os.listdir(class_dir)):
            path = os.path.join(class_dir, fname)
            img = Image.open(path).convert("RGB")
            img = img.resize(image_size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            xs.append(arr.reshape(-1))
            ys.append(class_to_idx[class_name])

    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    return X, y, class_names


def make_split_indices(
    n_samples: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }


def apply_split(
    X: np.ndarray,
    y: np.ndarray,
    split_indices: Dict[str, np.ndarray],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return {
        part: (X[idx], y[idx])
        for part, idx in split_indices.items()
    }


def standardize(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / std


def batch_iterator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        yield X[idx], y[idx]

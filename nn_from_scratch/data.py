from typing import Tuple, Iterator, Optional
import numpy as np


def make_spiral(n_classes: int = 3, points_per_class: int = 200, noise: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    N = points_per_class * n_classes
    X = np.zeros((N, 2))
    y = np.zeros(N, dtype=np.int64)
    for j in range(n_classes):
        ix = range(points_per_class * j, points_per_class * (j + 1))
        r = np.linspace(0.0, 1, points_per_class)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, points_per_class) + rng.randn(points_per_class) * noise  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y


class SpiralDataset:
    def __init__(self, n_classes: int = 3, points_per_class: int = 200, noise: float = 0.2, seed: int = 42):
        self.X, self.y = make_spiral(n_classes, points_per_class, noise, seed)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataLoader:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True, seed: Optional[int] = None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        N = self.X.shape[0]
        idxs = np.arange(N)
        if self.shuffle:
            self.rng.shuffle(idxs)
        for start in range(0, N, self.batch_size):
            end = start + self.batch_size
            batch_idx = idxs[start:end]
            yield self.X[batch_idx], self.y[batch_idx]


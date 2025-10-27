import math
import numpy as np
from typing import Optional


class RandomContext:
    """Manage RNG state and dtype.

    If no seed is provided, this context follows the global NumPy RNG
    (np.random), so calling np.random.seed(...) will also affect inits.
    If a seed is provided via constructor or set_seed, it uses its own
    independent RNG.
    """

    def __init__(self, seed: Optional[int] = None, dtype: np.dtype = np.float64):
        self.dtype = dtype
        self.use_global = seed is None
        self.rng = None if self.use_global else np.random.RandomState(seed)

    def set_seed(self, seed: Optional[int]):
        if seed is None:
            self.use_global = True
            self.rng = None
        else:
            self.use_global = False
            if self.rng is None:
                self.rng = np.random.RandomState(seed)
            else:
                self.rng.seed(seed)

    def randn(self, *shape):
        if self.use_global:
            arr = np.random.randn(*shape)
        else:
            arr = self.rng.randn(*shape)
        return arr.astype(self.dtype, copy=False)


# Global default random context
_GLOBAL_RC = RandomContext()


def set_seed(seed: Optional[int]):
    _GLOBAL_RC.set_seed(seed)


def set_default_dtype(dtype: np.dtype):
    _GLOBAL_RC.dtype = dtype


def get_default_dtype() -> np.dtype:
    return _GLOBAL_RC.dtype


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    Y = np.zeros((y.size, num_classes), dtype=get_default_dtype())
    Y[np.arange(y.size), y] = 1.0
    return Y


def he_init(fan_in: int, fan_out: int, rc: Optional[RandomContext] = None) -> np.ndarray:
    rc = rc or _GLOBAL_RC
    scale = math.sqrt(2.0 / fan_in)
    return rc.randn(fan_in, fan_out) * scale


def xavier_init(fan_in: int, fan_out: int, rc: Optional[RandomContext] = None) -> np.ndarray:
    rc = rc or _GLOBAL_RC
    scale = math.sqrt(1.0 / fan_in)
    return rc.randn(fan_in, fan_out) * scale

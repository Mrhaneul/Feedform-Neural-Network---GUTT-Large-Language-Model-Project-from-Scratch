import math
import numpy as np
from typing import Optional


class RandomContext:
    """Small helper to manage RNG state and dtype centrally.

    This keeps things slightly more OO without overcomplicating utilities.
    """

    def __init__(self, seed: Optional[int] = None, dtype: np.dtype = np.float32):
        self.dtype = dtype
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def set_seed(self, seed: int):
        self.rng.seed(seed)

    def randn(self, *shape):
        return self.rng.randn(*shape).astype(self.dtype, copy=False)


# Global default random context
_GLOBAL_RC = RandomContext()


def set_seed(seed: int):
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


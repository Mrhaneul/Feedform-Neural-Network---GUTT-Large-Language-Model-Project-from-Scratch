import numpy as np
from typing import List, Tuple, Iterable, Optional

from .utils import he_init, xavier_init


class Layer:
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params_and_grads(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        return []

    def train(self):
        pass

    def eval(self):
        pass


class Dense(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init: str = "xavier"):
        if init == "he":
            W = he_init(in_features, out_features)
        else:
            W = xavier_init(in_features, out_features)
        self.W = W
        self.b = np.zeros(out_features, dtype=self.W.dtype) if bias else None
        self.x: Optional[np.ndarray] = None
        # gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if self.b is not None else None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.x = x
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # grad_out: (N, out_features)
        self.dW[...] = self.x.T @ grad_out  # (in_features, out_features)
        if self.b is not None:
            self.db[...] = np.sum(grad_out, axis=0)  # (out_features,)
        grad_x = grad_out @ self.W.T  # (N, in_features)
        return grad_x

    def params_and_grads(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if self.b is None:
            return [(self.W, self.dW)]
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU(Layer):
    def __init__(self):
        self.mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self.mask


class Sigmoid(Layer):
    def __init__(self):
        self.out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        # stable sigmoid
        out = np.empty_like(x)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1 / (1 + np.exp(-x[pos]))
        expx = np.exp(x[neg])
        out[neg] = expx / (1 + expx)
        self.out = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self.out * (1 - self.out)


class Tanh(Layer):
    def __init__(self):
        self.out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (1 - self.out ** 2)


class Softmax(Layer):
    def __init__(self):
        self.out: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        # x: (N, C)
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x_shift)
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # Not typically used standalone with CE; use SoftmaxCrossEntropy
        N, C = grad_out.shape
        grad_in = np.empty_like(grad_out)
        for i in range(N):
            y = self.out[i].reshape(-1, 1)  # (C,1)
            J = np.diagflat(y) - y @ y.T    # (C,C)
            grad_in[i] = J @ grad_out[i]
        return grad_in

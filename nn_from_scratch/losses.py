import numpy as np
from typing import Optional


class Loss:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class MSE(Loss):
    def __init__(self):
        self.y_pred: Optional[np.ndarray] = None
        self.y_true: Optional[np.ndarray] = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        return float(0.5 * np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        N = self.y_pred.shape[0]
        return (self.y_pred - self.y_true) / N


class SoftmaxCrossEntropy(Loss):
    # Combines stable softmax + cross-entropy for integer labels or one-hot labels.
    def __init__(self):
        self.probs: Optional[np.ndarray] = None
        self.y_true: Optional[np.ndarray] = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        # logits: (N, C), y_true: (N,) int or (N,C) one-hot
        x = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(x)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.probs = probs
        if y_true.ndim == 1:
            # integer labels
            N = logits.shape[0]
            loss = -np.log(np.clip(probs[np.arange(N), y_true], 1e-12, 1.0))
            self.y_true = y_true
            return float(np.mean(loss))
        else:
            # one-hot
            self.y_true = y_true
            loss = -np.sum(y_true * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
            return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        N = self.probs.shape[0]
        if self.y_true.ndim == 1:
            grad = self.probs.copy()
            grad[np.arange(N), self.y_true] -= 1.0
            grad /= N
            return grad
        else:
            grad = (self.probs - self.y_true) / N
            return grad


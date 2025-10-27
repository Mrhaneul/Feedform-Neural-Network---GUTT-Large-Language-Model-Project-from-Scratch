import numpy as np
from typing import Dict, Tuple, Iterable


class Optimizer:
    def step(self, params_and_grads: Iterable[Tuple[np.ndarray, np.ndarray]]):
        raise NotImplementedError

    def zero_grad(self, params_and_grads: Iterable[Tuple[np.ndarray, np.ndarray]]):
        for _, g in params_and_grads:
            g[...] = 0.0


class SGD(Optimizer):
    def __init__(self, lr: float = 1e-2, momentum: float = 0.0, weight_decay: float = 0.0, grad_clip=None):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.velocities: Dict[int, np.ndarray] = {}

    def step(self, params_and_grads: Iterable[Tuple[np.ndarray, np.ndarray]]):
        for idx, (p, g) in enumerate(params_and_grads):
            gg = g
            if self.weight_decay != 0.0:
                gg = gg + self.weight_decay * p
            if self.grad_clip is not None:
                gg = np.clip(gg, -self.grad_clip, self.grad_clip)
            v = self.velocities.get(idx, np.zeros_like(p))
            v = self.momentum * v - self.lr * gg
            p += v
            self.velocities[idx] = v


class Adam(Optimizer):
    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.0, grad_clip=None):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, params_and_grads: Iterable[Tuple[np.ndarray, np.ndarray]]):
        self.t += 1
        for idx, (p, g) in enumerate(params_and_grads):
            gg = g
            if self.weight_decay != 0.0:
                gg = gg + self.weight_decay * p
            if self.grad_clip is not None:
                gg = np.clip(gg, -self.grad_clip, self.grad_clip)
            m = self.m.get(idx, np.zeros_like(p))
            v = self.v.get(idx, np.zeros_like(p))
            m = self.b1 * m + (1 - self.b1) * gg
            v = self.b2 * v + (1 - self.b2) * (gg * gg)
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p += -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.m[idx] = m
            self.v[idx] = v


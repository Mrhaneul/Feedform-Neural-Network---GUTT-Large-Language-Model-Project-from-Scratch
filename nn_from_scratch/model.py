import numpy as np
from typing import Iterable, Tuple, List

from .layers import Layer


class Sequential:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self._training = True

    # modes
    def train(self):
        self._training = True
        for l in self.layers:
            l.train()

    def eval(self):
        self._training = False
        for l in self.layers:
            l.eval()

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        # Prefer explicit training flag, else internal mode
        train_flag = training if training is not None else self._training
        for layer in self.layers:
            x = layer.forward(x, training=train_flag)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        for layer in self.layers:
            for p, g in layer.params_and_grads():
                yield p, g

    # persistence
    def state_dict(self) -> dict:
        state = {}
        for i, l in enumerate(self.layers):
            for j, (p, _) in enumerate(l.params_and_grads()):
                state[f"layer{i}.param{j}"] = p
        return state

    def load_state_dict(self, state: dict):
        for i, l in enumerate(self.layers):
            for j, (p, _) in enumerate(l.params_and_grads()):
                key = f"layer{i}.param{j}"
                if key not in state:
                    raise KeyError(f"Missing key in state: {key}")
                src = state[key]
                if src.shape != p.shape:
                    raise ValueError(f"Shape mismatch for {key}: {src.shape} vs {p.shape}")
                p[...] = src

    def save(self, path: str):
        state = self.state_dict()
        np.savez(path, **state)

    @classmethod
    def load(cls, path: str, template_model: "Sequential") -> "Sequential":
        data = np.load(path)
        state = {k: data[k] for k in data.files}
        template_model.load_state_dict(state)
        return template_model


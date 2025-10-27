from typing import Callable, Dict, List, Optional
import numpy as np

from .model import Sequential


class Callback:
    def on_epoch_end(self, epoch: int, history: Dict[str, List[float]]):
        pass


class Trainer:
    def __init__(self, model: Sequential, loss_fn, optimizer, batch_size: int = 64, shuffle: bool = True,
                 callbacks: Optional[List[Callback]] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.callbacks = callbacks or []

    def iterate_minibatches(self, X: np.ndarray, y: np.ndarray):
        N = X.shape[0]
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        for start in range(0, N, self.batch_size):
            end = start + self.batch_size
            batch_idx = idxs[start:end]
            yield X[batch_idx], y[batch_idx]

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 100, verbose: bool = True):
        history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            n_samples = 0

            for Xb, yb in self.iterate_minibatches(X, y):
                logits = self.model.forward(Xb, training=True)
                loss = self.loss_fn.forward(logits, yb)
                grad = self.loss_fn.backward()
                self.model.backward(grad)
                self.optimizer.step(self.model.params_and_grads())
                epoch_loss += loss * Xb.shape[0]
                if logits.shape[1] > 1:
                    preds = np.argmax(logits, axis=1)
                    if yb.ndim > 1:
                        y_true = np.argmax(yb, axis=1)
                    else:
                        y_true = yb
                    correct += np.sum(preds == y_true)
                n_samples += Xb.shape[0]

            epoch_loss /= max(1, n_samples)
            history["loss"].append(epoch_loss)

            if n_samples and logits.shape[1] > 1:
                history["acc"].append(correct / n_samples)
            else:
                history["acc"].append(None)

            # validation
            self.model.eval()
            if X_val is not None and y_val is not None:
                val_logits = self.model.forward(X_val, training=False)
                val_loss = self.loss_fn.forward(val_logits, y_val)
                history["val_loss"].append(val_loss)
                if val_logits.shape[1] > 1:
                    val_preds = np.argmax(val_logits, axis=1)
                    if y_val.ndim > 1:
                        val_true = np.argmax(y_val, axis=1)
                    else:
                        val_true = y_val
                    val_acc = float(np.mean(val_preds == val_true))
                    history["val_acc"].append(val_acc)
                else:
                    history["val_acc"].append(None)
            else:
                history["val_loss"].append(None)
                history["val_acc"].append(None)

            # callbacks
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, history)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                acc_str = f" acc={history['acc'][-1]:.3f}" if history['acc'][-1] is not None else ""
                val_str = ""
                if history["val_loss"][-1] is not None:
                    val_acc_str = f", val_acc={history['val_acc'][-1]:.3f}" if history['val_acc'][-1] is not None else ""
                    val_str = f", val_loss={history['val_loss'][-1]:.4f}{val_acc_str}"
                print(f"Epoch {epoch:4d}/{epochs} - loss={epoch_loss:.4f}{acc_str}{val_str}")

        return history


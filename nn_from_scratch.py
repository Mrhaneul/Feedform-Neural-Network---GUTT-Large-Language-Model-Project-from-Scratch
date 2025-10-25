#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Feedforward Neural Network from scratch (NumPy-only)
# Features:
# - Dense layers with Xavier/He init
# - Activations: ReLU, Sigmoid, Tanh, Softmax
# - Losses: MSE, CrossEntropy (w/ stable Softmax+CE combo)
# - Optimizers: SGD (w/ momentum), Adam
# - Mini-batch training, L2 weight decay, gradient clipping
# - Simple Trainer
# - Demo on a 3-class spiral dataset
#
# Run:  python nn_from_scratch.py

import math
import numpy as np

# -------------------------- Utility --------------------------

def one_hot(y, num_classes):
    Y = np.zeros((y.size, num_classes))
    Y[np.arange(y.size), y] = 1.0
    return Y

def he_init(fan_in, fan_out):
    scale = math.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * scale

def xavier_init(fan_in, fan_out):
    scale = math.sqrt(1.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * scale


# -------------------------- Layers --------------------------

class Layer:
    def forward(self, x, training=True):
        raise NotImplementedError
    def backward(self, grad_out):
        raise NotImplementedError
    def params_and_grads(self):
        return []

class Dense(Layer):
    def __init__(self, in_features, out_features, bias=True, init="xavier"):
        if init == "he":
            W = he_init(in_features, out_features)
        else:
            W = xavier_init(in_features, out_features)
        self.W = W
        self.b = np.zeros(out_features) if bias else None
        self.x = None
        # gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if self.b is not None else None

    def forward(self, x, training=True):
        self.x = x
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def backward(self, grad_out):
        # grad_out: (N, out_features)
        self.dW[...] = self.x.T @ grad_out  # (in_features, out_features)
        if self.b is not None:
            self.db[...] = np.sum(grad_out, axis=0)  # (out_features,)
        grad_x = grad_out @ self.W.T  # (N, in_features)
        return grad_x

    def params_and_grads(self):
        if self.b is None:
            return [(self.W, self.dW)]
        return [(self.W, self.dW), (self.b, self.db)]


# -------------------------- Activations --------------------------

class ReLU(Layer):
    def __init__(self):
        self.mask = None
    def forward(self, x, training=True):
        self.mask = x > 0
        return np.maximum(0, x)
    def backward(self, grad_out):
        return grad_out * self.mask

class Sigmoid(Layer):
    def __init__(self):
        self.out = None
    def forward(self, x, training=True):
        # stable sigmoid
        out = np.empty_like(x)
        pos = x >= 0
        neg = ~pos
        out[pos] = 1 / (1 + np.exp(-x[pos]))
        expx = np.exp(x[neg])
        out[neg] = expx / (1 + expx)
        self.out = out
        return out
    def backward(self, grad_out):
        return grad_out * self.out * (1 - self.out)

class Tanh(Layer):
    def __init__(self):
        self.out = None
    def forward(self, x, training=True):
        self.out = np.tanh(x)
        return self.out
    def backward(self, grad_out):
        return grad_out * (1 - self.out**2)

class Softmax(Layer):
    def __init__(self):
        self.out = None
    def forward(self, x, training=True):
        # x: (N, C)
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x_shift)
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out
    def backward(self, grad_out):
        # Not typically used standalone with CE; use SoftmaxCrossEntropy
        N, C = grad_out.shape
        grad_in = np.empty_like(grad_out)
        for i in range(N):
            y = self.out[i].reshape(-1, 1)  # (C,1)
            J = np.diagflat(y) - y @ y.T    # (C,C)
            grad_in[i] = J @ grad_out[i]
        return grad_in


# -------------------------- Losses --------------------------

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError

class MSE(Loss):
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    def forward(self, y_pred, y_true):
        # y_true can be continuous or one-hot
        self.y_pred = y_pred
        self.y_true = y_true
        return 0.5 * np.mean((y_pred - y_true) ** 2)
    def backward(self):
        N = self.y_pred.shape[0]
        return (self.y_pred - self.y_true) / N

class SoftmaxCrossEntropy(Loss):
    # Combines stable softmax + cross-entropy for integer labels or one-hot labels.
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
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
            return np.mean(loss)
        else:
            # one-hot
            self.y_true = y_true
            loss = -np.sum(y_true * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
            return np.mean(loss)

    def backward(self):
        N = self.probs.shape[0]
        if self.y_true.ndim == 1:
            grad = self.probs.copy()
            grad[np.arange(N), self.y_true] -= 1.0
            grad /= N
            return grad
        else:
            grad = (self.probs - self.y_true) / N
            return grad


# -------------------------- Model --------------------------

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self):
        for layer in self.layers:
            for p, g in layer.params_and_grads():
                yield p, g


# -------------------------- Optimizers --------------------------

class Optimizer:
    def step(self, params_and_grads):
        raise NotImplementedError
    def zero_grad(self, params_and_grads):
        for _, g in params_and_grads:
            g[...] = 0.0

class SGD(Optimizer):
    def __init__(self, lr=1e-2, momentum=0.0, weight_decay=0.0, grad_clip=None):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.velocities = {}

    def step(self, params_and_grads):
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
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip=None):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params_and_grads):
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


# -------------------------- Training helpers --------------------------

class Trainer:
    def __init__(self, model, loss_fn, optimizer, batch_size=64, shuffle=True):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.shuffle = shuffle

    def iterate_minibatches(self, X, y):
        N = X.shape[0]
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        for start in range(0, N, self.batch_size):
            end = start + self.batch_size
            batch_idx = idxs[start:end]
            yield X[batch_idx], y[batch_idx]

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, verbose=True):
        history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            correct = 0
            n_samples = 0

            for Xb, yb in self.iterate_minibatches(X, y):
                # forward
                logits = self.model.forward(Xb, training=True)
                loss = self.loss_fn.forward(logits, yb)
                # backward
                grad = self.loss_fn.backward()
                self.model.backward(grad)
                # update
                self.optimizer.step(self.model.params_and_grads())
                # track
                epoch_loss += loss * Xb.shape[0]
                if logits.shape[1] > 1:
                    preds = np.argmax(logits, axis=1)
                    if yb.ndim > 1:
                        y_true = np.argmax(yb, axis=1)
                    else:
                        y_true = yb
                    correct += np.sum(preds == y_true)
                n_samples += Xb.shape[0]

            epoch_loss /= n_samples
            history["loss"].append(epoch_loss)

            if n_samples and logits.shape[1] > 1:
                history["acc"].append(correct / n_samples)
            else:
                history["acc"].append(None)

            # validation
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
                    val_acc = np.mean(val_preds == val_true)
                    history["val_acc"].append(val_acc)
                else:
                    history["val_acc"].append(None)
            else:
                history["val_loss"].append(None)
                history["val_acc"].append(None)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                acc_str = f" acc={history['acc'][-1]:.3f}" if history['acc'][-1] is not None else ""
                val_str = ""
                if history["val_loss"][-1] is not None:
                    val_acc_str = f", val_acc={history['val_acc'][-1]:.3f}" if history['val_acc'][-1] is not None else ""
                    val_str = f", val_loss={history['val_loss'][-1]:.4f}{val_acc_str}"
                print(f"Epoch {epoch:4d}/{epochs} - loss={epoch_loss:.4f}{acc_str}{val_str}")

        return history


# -------------------------- Demo: spiral classification --------------------------

def make_spiral(n_classes=3, points_per_class=200, noise=0.2, seed=42):
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

def demo():
    np.random.seed(0)
    X, y = make_spiral(n_classes=3, points_per_class=300, noise=0.2, seed=0)

    model = Sequential([
        Dense(2, 64, init="he"),
        ReLU(),
        Dense(64, 64, init="he"),
        ReLU(),
        Dense(64, 3, init="xavier"),
    ])

    loss = SoftmaxCrossEntropy()
    opt = Adam(lr=1e-2, weight_decay=1e-4)

    # split train/val
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    trainer = Trainer(model, loss, opt, batch_size=128, shuffle=True)
    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=200, verbose=True)

    # final accuracy
    logits = model.forward(X_val, training=False)
    preds = np.argmax(logits, axis=1)
    acc = np.mean(preds == y_val)
    print(f"Final validation accuracy: {acc:.3f}")

if __name__ == "__main__":
    demo()

#!/usr/bin/env python3
import numpy as np

from ..model import Sequential
from ..layers import Dense, ReLU
from ..losses import SoftmaxCrossEntropy
from ..optimizers import Adam
from ..trainer import Trainer
from ..data import make_spiral


def main():
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
    main()


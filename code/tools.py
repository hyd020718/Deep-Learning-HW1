import json
import os
import random
from typing import Any
import numpy as np
import math


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_python(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_python(obj), f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Linear:
    def __init__(self, in_dim: int, out_dim: int):
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        self.W = np.random.uniform(-limit, limit, size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros((1, out_dim), dtype=np.float32)

        self.x_cache = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_cache = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self.x_cache
        self.dW = x.T @ grad_out
        self.db = grad_out.sum(axis=0, keepdims=True)
        grad_x = grad_out @ self.W.T
        return grad_x


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self.mask


class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * (1.0 - self.out ** 2)


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
        return self.out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self.out * (1.0 - self.out)


def make_activation(name: str):
    name = name.lower()
    if name == "relu":
        return ReLU()
    if name == "tanh":
        return Tanh()
    if name == "sigmoid":
        return Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


class SGD:
    def __init__(self, parameters_and_grads_fn, lr: float):
        self.parameters_and_grads_fn = parameters_and_grads_fn
        self.lr = lr

    def step(self) -> None:
        for param, grad in self.parameters_and_grads_fn():
            param -= self.lr * grad

    def set_lr(self, lr: float) -> None:
        self.lr = lr


class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.targets = None
        self.logits = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        self.probs = probs
        self.targets = targets
        self.logits = logits

        n = logits.shape[0]
        loss = -np.log(probs[np.arange(n), targets] + 1e-12).mean()
        return float(loss)

    def backward(self) -> np.ndarray:
        n = self.logits.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.targets] -= 1.0
        grad = grad / n
        return grad
    

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm



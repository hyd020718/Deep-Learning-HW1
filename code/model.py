from typing import Dict, List, Tuple

import numpy as np

from .tools import Linear, make_activation


class MLP3:
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        num_classes: int,
        activation: str,
    ):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes
        self.activation_name = activation

        self.fc1 = Linear(input_dim, hidden_dim1)
        self.act1 = make_activation(activation)
        self.fc2 = Linear(hidden_dim1, hidden_dim2)
        self.act2 = make_activation(activation)
        self.fc3 = Linear(hidden_dim2, num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = self.fc1.forward(x)
        a1 = self.act1.forward(z1)
        z2 = self.fc2.forward(a1)
        a2 = self.act2.forward(z2)
        out = self.fc3.forward(a2)
        return out

    def backward(self, grad_out: np.ndarray) -> None:
        grad = self.fc3.backward(grad_out)
        grad = self.act2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.act1.backward(grad)
        _ = self.fc1.backward(grad)

    def parameters_and_grads(self):
        return [
            (self.fc1.W, self.fc1.dW),
            (self.fc1.b, self.fc1.db),
            (self.fc2.W, self.fc2.dW),
            (self.fc2.b, self.fc2.db),
            (self.fc3.W, self.fc3.dW),
            (self.fc3.b, self.fc3.db),
        ]

    def l2_penalty(self) -> float:
        return 0.5 * (
            np.sum(self.fc1.W ** 2)
            + np.sum(self.fc2.W ** 2)
            + np.sum(self.fc3.W ** 2)
        )

    def add_l2_grads(self, weight_decay: float) -> None:
        self.fc1.dW += weight_decay * self.fc1.W
        self.fc2.dW += weight_decay * self.fc2.W
        self.fc3.dW += weight_decay * self.fc3.W

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def save(self, path: str, metadata: Dict) -> None:
        np.savez(
            path,
            fc1_W=self.fc1.W,
            fc1_b=self.fc1.b,
            fc2_W=self.fc2.W,
            fc2_b=self.fc2.b,
            fc3_W=self.fc3.W,
            fc3_b=self.fc3.b,
            input_dim=np.array(self.input_dim),
            hidden_dim1=np.array(self.hidden_dim1),
            hidden_dim2=np.array(self.hidden_dim2),
            num_classes=np.array(self.num_classes),
            activation=np.array(self.activation_name),
            **metadata,
        )

    def load_weights(self, ckpt: Dict[str, np.ndarray]) -> None:
        self.fc1.W = ckpt["fc1_W"]
        self.fc1.b = ckpt["fc1_b"]
        self.fc2.W = ckpt["fc2_W"]
        self.fc2.b = ckpt["fc2_b"]
        self.fc3.W = ckpt["fc3_W"]
        self.fc3.b = ckpt["fc3_b"]

    @staticmethod
    def from_checkpoint(path: str) -> Tuple["MLP3", Dict]:
        ckpt = np.load(path, allow_pickle=True)
        model = MLP3(
            input_dim=int(ckpt["input_dim"]),
            hidden_dim1=int(ckpt["hidden_dim1"]),
            hidden_dim2=int(ckpt["hidden_dim2"]),
            num_classes=int(ckpt["num_classes"]),
            activation=str(ckpt["activation"].item() if hasattr(ckpt["activation"], "item") else ckpt["activation"]),
        )
        model.load_weights(ckpt)
        metadata = {
            "class_names": ckpt["class_names"].tolist(),
            "image_size": tuple(int(v) for v in ckpt["image_size"].tolist()),
        }
        return model, metadata

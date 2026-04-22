import math
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
from .model import MLP3
from .tools import ensure_dir


def plot_training_curves(history: Dict[str, List[float]], save_dir: str) -> None:
    ensure_dir(save_dir)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_acc"], label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_accuracy.png"), dpi=200)
    plt.close()



def plot_confusion_matrix(cm, class_names, save_path):
    red_white_cmap = LinearSegmentedColormap.from_list(
        "red_white_soft",
        ["#FFF9F5","#F7D9DA","#EFA3A8","#C94F5C","#8B1E3F"]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap=red_white_cmap, aspect="auto")

    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "#4A2A2A"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def visualize_first_layer_weights(
    model_path: str,
    image_size=(64, 64),
    max_neurons=16,
    save_dir="first_layer_weights",
):
    ckpt = np.load(model_path, allow_pickle=True)
    W1 = ckpt["fc1_W"]

    H, W_img = image_size
    C = 3
    input_dim, hidden_dim = W1.shape

    os.makedirs(save_dir, exist_ok=True)

    n_show = min(hidden_dim, max_neurons)

    for i in range(n_show):
        w = W1[:, i].reshape(H, W_img, C)

        w_min, w_max = w.min(), w.max()
        w_vis = (w - w_min) / (w_max - w_min + 1e-8)

        plt.figure(figsize=(4, 4))
        plt.imshow(w_vis)
        plt.title(f"Neuron {i}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"neuron_{i:03d}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

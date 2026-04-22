from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass
class ExperimentConfig:
    data_root: str = "EuroSAT_RGB"
    image_size: Tuple[int, int] = (64, 64)
    num_classes: int = 10
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42

    hidden_dim1: int = 256
    hidden_dim2: int = 128
    activation: str = "relu"

    batch_size: int = 128
    epochs: int = 30
    lr: float = 1e-2
    lr_decay: float = 0.95
    weight_decay: float = 1e-4

    output_dir: str = "outputs/default_run"
    checkpoint_name: str = "best_model.npz"

    def to_dict(self) -> Dict:
        return asdict(self)

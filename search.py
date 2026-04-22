from code.config import ExperimentConfig
from code.trainer import grid_search


if __name__ == "__main__":
    base_config = ExperimentConfig(
        data_root="EuroSAT_RGB",
        image_size=(64, 64),
        num_classes=10,
        epochs=10,
        output_dir="outputs/search_runs",
    )

    search_space = {
        "batch_size": [64, 128],
        "lr": [5e-3, 1e-2],
        "lr_decay": [0.95, 0.99],
        "hidden_dim1": [128, 256],
        "hidden_dim2": [64, 128],
        "weight_decay": [5e-5, 5e-4],
        "activation": ["relu","tanh","sigmoid"],
    }
    results = grid_search(base_config, search_space)

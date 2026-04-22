import argparse
import os
from eurosat_mlp.tools import load_json
from eurosat_mlp.visualize import visualize_first_layer_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize first-layer hidden weights.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--max_neurons", type=int, default=36)
    args = parser.parse_args()

    config = load_json(os.path.join(args.run_dir, "config.json"))
    model_path = os.path.join(args.run_dir, config.get("checkpoint_name", "best_model.npz"))
    save_path = os.path.join(args.run_dir, "plots", "first_layer_weights_manual.png")

    visualize_first_layer_weights(
        model_path=model_path,
        save_path=save_path,
        image_size=tuple(config.get("image_size", [64, 64])),
        max_neurons=args.max_neurons,
    )
    print("Saved to:", save_path)

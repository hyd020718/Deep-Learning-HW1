import argparse
from eurosat_mlp.config import ExperimentConfig
from eurosat_mlp.trainer import train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a 3-layer MLP on EuroSAT.")
    parser.add_argument("--data_root", type=str, default="EuroSAT_RGB")
    parser.add_argument("--image_size", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=718)

    parser.add_argument("--hidden_dim1", type=int, default=256)
    parser.add_argument("--hidden_dim2", type=int, default=128)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "sigmoid"])

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--output_dir", type=str, default="outputs/default_run")
    parser.add_argument("--checkpoint_name", type=str, default="best_model.npz")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    config = ExperimentConfig(
        data_root=args.data_root,
        image_size=tuple(args.image_size),
        num_classes=args.num_classes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        activation=args.activation,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name,
    )
    train_experiment(config)

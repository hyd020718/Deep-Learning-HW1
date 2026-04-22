import argparse

from eurosat_mlp.trainer import evaluate_from_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the saved best model on the test split.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_subdir", type=str, default="eval")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate_from_run_dir(
        run_dir=args.run_dir,
        batch_size=args.batch_size,
        output_subdir=args.output_subdir,
    )

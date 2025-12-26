import argparse

from vit_train import train
from inference import run_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "infer"],
        help="run mode"
    )

    args = parser.parse_args()
    print("mode:", args.mode)

    if args.mode == "train":
        train()

    elif args.mode == "infer":
        run_inference()


if __name__ == "__main__":
    main()

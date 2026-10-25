"""Usage: uv run python train.py --config configs/baseline.yaml [a.b=value ...]"""
import argparse

from dcase_ae.config import load_config
from dcase_ae.trainer import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. data.machine=ToyCar train.steps=2000")
    args = ap.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()

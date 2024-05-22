"""Train trait models using the given configuration."""

import argparse
from box import ConfigBox
from src.conf.conf import get_config
from src.models import autogluon


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train AutoGluon model")
    parser.add_argument("-s", "--sample", type=float, default=1.0, help="Sample size")
    return parser.parse_args()


def main(args: argparse.Namespace, cfg: ConfigBox = get_config()) -> None:
    """Train a set of models using the given configuration."""
    if cfg.train.arch == "autogluon":
        autogluon.train(cfg, args.sample)
    else:
        raise ValueError(f"Unknown architecture: {cfg.train.arch}")


if __name__ == "__main__":
    main(cli())

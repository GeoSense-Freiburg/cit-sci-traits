"""Train trait models using the given configuration."""

import argparse

from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import activate_env
from src.models import autogluon


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train AutoGluon model")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-s", "--sample", type=float, default=1.0, help="Sample size")
    parser.add_argument("-r", "--resume", action="store_true", help="Resume training")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Dry run")
    return parser.parse_args()


def main(args: argparse.Namespace, cfg: ConfigBox = get_config()) -> None:
    """Train a set of models using the given configuration."""
    activate_env()
    if cfg.train.arch == "autogluon":
        autogluon.train_models(args.sample, args.debug, args.resume, args.dry_run)
    else:
        raise ValueError(f"Unknown architecture: {cfg.train.arch}")


if __name__ == "__main__":
    main(cli())

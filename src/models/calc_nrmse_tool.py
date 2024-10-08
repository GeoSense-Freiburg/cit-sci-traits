"""Calculate the normalized root mean squared error for each trait model and update the
models' evaluation results accordingly. Only a one-time operation as nRMSE has
now been implemented in the training pipeline."""

import argparse
from pathlib import Path

import pandas as pd
from box import ConfigBox
from tqdm import tqdm

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import (
    get_cv_splits_dir,
    get_latest_run,
    get_models_dir,
    get_y_fn,
)


def calc_nrmse(rmse: float, norm_factor: float) -> float:
    """Calculate the normalized root mean squared error."""
    return rmse / norm_factor


def add_nrmse(val: pd.Series, eval_results: pd.DataFrame) -> pd.DataFrame:
    """Add the normalized root mean squared error to the evaluation results."""
    norm_factor = val.quantile(0.99) - val.quantile(0.01)
    nrmse = calc_nrmse(eval_results["root_mean_squared_error"].values[0], norm_factor)
    return eval_results.assign(norm_root_mean_squared_error=nrmse)


def safe_overwrite(fn: Path, df: pd.DataFrame) -> None:
    """Safely overwrite a file."""
    backup_fn = fn.with_suffix(".bak")
    try:
        fn.rename(backup_fn)
        df.to_csv(fn)
        backup_fn.unlink()
    except Exception:
        if backup_fn.exists():
            backup_fn.rename(fn)
        raise


def update_fold_eval_results(val: pd.Series, results_fn: Path) -> None:
    """Update the evaluation results for a given fold."""
    eval_results = pd.read_csv(results_fn, index_col=0)
    eval_results = add_nrmse(val, eval_results)
    try:
        eval_results.to_csv(results_fn)
    except PermissionError:
        safe_overwrite(results_fn, eval_results)


def reaggregate_results(fold_results: list[Path]) -> pd.DataFrame:
    """Reaggregate the evaluation results for all folds."""
    return (
        pd.concat(
            [pd.read_csv(fold_result, index_col=0) for fold_result in fold_results],
        )
        .drop(columns=["fold"])
        .reset_index(names="index")
        .groupby("index")
        .agg(["mean", "std"])
    )


def process_trait_set(
    ts_dir: Path, trait_df: pd.DataFrame, eval_results_fn: str
) -> None:
    """Process the evaluation results for a given trait set within a trait."""
    fold_results = list(ts_dir.glob(f"cv/fold_*/{eval_results_fn}"))
    trait_id = trait_df.columns.difference(["fold"]).values[0]

    for fold_result in fold_results:
        fold = int(fold_result.parent.stem.split("_")[-1])
        val = trait_df.query(f"fold == {fold}")[trait_id]
        update_fold_eval_results(val, fold_result)

    trait_results = reaggregate_results(fold_results)
    try:
        trait_results.to_csv(ts_dir / f"{eval_results_fn}")
    except PermissionError:
        safe_overwrite(ts_dir / f"{eval_results_fn}", trait_results)


def process_trait(run_dir: Path, trait_df: pd.DataFrame, eval_results_fn: str) -> None:
    """Process the evaluation results for a given trait."""
    ts_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
    for trait_dir in ts_dirs:
        process_trait_set(trait_dir, trait_df, eval_results_fn)


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate the normalized root mean squared error."
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode.")
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()):
    """Calculate the normalized root mean squared error for each trait model and update the
    models' evaluation results accordingly."""
    models_dir = get_models_dir() / "debug" if args.debug else get_models_dir()
    trait_dirs = [d for d in models_dir.iterdir() if d.is_dir()]

    log.info("Reading Y data...")
    all_y = (
        pd.read_parquet(get_y_fn())
        .query("source == 's'")
        .drop(columns=["source"])
        .set_index(["y", "x"])
    )

    for trait_dir in tqdm(trait_dirs, total=len(trait_dirs)):
        log.info("Processing trait: %s", trait_dir)
        log.info("Joining Y data with CV splits...")
        trait_df = all_y.join(
            pd.read_parquet(
                get_cv_splits_dir() / f"{trait_dir.stem}.parquet"
            ).set_index(["y", "x"]),
            how="inner",
            validate="1:1",
        )[[trait_dir.stem, "fold"]]

        last_run = get_latest_run(trait_dir / cfg.train.arch)
        process_trait(last_run, trait_df, cfg.train.eval_results)


if __name__ == "__main__":
    main()

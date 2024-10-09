import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from box import ConfigBox
from dask import compute, delayed
from sklearn import metrics

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import (
    get_cv_splits_dir,
    get_latest_run,
    get_models_dir,
    get_y_fn,
)
from src.utils.spatial_utils import lat_weights, weighted_pearson_r


def trait_with_splits(all_y: pd.DataFrame, trait_id: str) -> pd.DataFrame:
    return all_y.join(
        pd.read_parquet(get_cv_splits_dir() / f"{trait_id}.parquet").set_index(
            ["y", "x"]
        ),
        how="inner",
        validate="1:1",
    )[[trait_id, "fold"]]


@delayed
def process_fold(fold_dir: Path, obs_df: pd.DataFrame) -> pd.DataFrame:
    predictor = TabularPredictor.load(str(fold_dir))
    pred = pd.DataFrame({"pred": predictor.predict(obs_df)})
    return pd.concat([obs_df, pd.DataFrame(pred)], axis=1).set_index(["y", "x"])


def get_stats(
    cv_obs_vs_pred: pd.DataFrame, resolution: int | float, log: bool = False
) -> dict[str, Any]:
    obs = cv_obs_vs_pred.obs.to_numpy()
    pred = cv_obs_vs_pred.pred.to_numpy()

    if log:
        obs = np.log(obs)
        pred = np.log(pred)
        cv_obs_vs_pred = cv_obs_vs_pred.assign(obs=obs, pred=pred)

    r2 = metrics.r2_score(obs, pred)
    pearsonr = cv_obs_vs_pred[["obs", "pred"]].corr().iloc[0, 1]
    pearsonr_wt = weighted_pearson_r(
        cv_obs_vs_pred,
        lat_weights(cv_obs_vs_pred.y.unique(), resolution),
    )
    root_mean_squared_error = metrics.root_mean_squared_error(obs, pred)
    norm_root_mean_squared_error = root_mean_squared_error / (
        cv_obs_vs_pred.obs.quantile(0.99) - cv_obs_vs_pred.obs.quantile(0.01)
    )
    mean_squared_error = np.mean((obs - pred) ** 2)
    mean_absolute_error = metrics.mean_absolute_error(obs, pred)
    median_absolute_error = metrics.median_absolute_error(obs, pred)

    return {
        "r2": r2,
        "pearsonr": pearsonr,
        "pearsonr_wt": pearsonr_wt,
        "root_mean_squared_error": root_mean_squared_error,
        "norm_root_mean_squared_error": norm_root_mean_squared_error,
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "median_absolute_error": median_absolute_error,
    }


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode.")
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """
    Main function for generating spatial CV trait statistics.
    """
    # TODO: Actually get feature data for splot Y values
    log.info("Reading Y data...")
    all_y = (
        pd.read_parquet(get_y_fn())
        .query("source == 's'")
        .drop(columns=["source"])
        .set_index(["y", "x"])
    )
    models_dir = get_models_dir() / "debug" if args.debug else get_models_dir()

    for trait_dir in models_dir.iterdir():
        if not trait_dir.is_dir():
            continue

        log.info("Processing trait: %s", trait_dir.stem)
        log.info("Joining Y data with CV splits...")
        trait_df = trait_with_splits(all_y, trait_dir.stem)

        latest_run = get_latest_run(trait_dir / cfg.train.arch)

        for ts_dir in latest_run.iterdir():
            if not ts_dir.is_dir():
                continue

            coll = []
            fold_dirs = [d for d in Path(ts_dir, "cv").iterdir() if d.is_dir()]
            fold_ids = [fold_dir.stem.split("_")[-1] for fold_dir in fold_dirs]

            log.info("Splitting folds...")
            fold_dfs = [
                trait_df.query(f"fold == {fold_id}")
                .drop(columns=["fold"])
                .reset_index()
                for fold_id in fold_ids
            ]

            delayed_results = [
                process_fold(fold_dir, obs_df)
                for fold_dir, obs_df in zip(fold_dirs, fold_dfs)
            ]
            log.info("Computing delayed results...")
            coll = compute(*delayed_results)

            log.info("Concatenating results...")
            cv_obs_vs_pred = pd.concat(coll).reset_index()
            log.info("Writing results to disk...")
            cv_obs_vs_pred.to_parquet(Path(ts_dir, "cv_obs_vs_pred.parquet"))

            log.info("Calculating stats...")
            stats = get_stats(cv_obs_vs_pred, cfg.target_resolution)
            stats_ln = get_stats(cv_obs_vs_pred, cfg.target_resolution, log=True)

            log.info("Writing stats to disk...")
            pd.DataFrame(stats, index=[0]).to_csv(Path(ts_dir, "cv_stats.parquet"))

            pd.DataFrame(stats_ln, index=[0]).to_csv(
                Path(ts_dir, "cv_stats_ln.parquet")
            )


if __name__ == "__main__":
    main()

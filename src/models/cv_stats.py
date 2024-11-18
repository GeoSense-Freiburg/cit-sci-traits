import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from box import ConfigBox
from dask import compute, delayed
from sklearn import metrics

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import (
    get_cv_splits_dir,
    get_latest_run,
    get_models_dir,
    get_power_transformer_fn,
    get_predict_imputed_fn,
    get_predict_mask_fn,
    get_y_fn,
)
from src.utils.spatial_utils import lat_weights, weighted_pearson_r
from src.utils.stat_utils import yeo_johnson_inverse_transform, yeo_johnson_transform
from src.utils.trait_utils import get_trait_number_from_id

TMP_DIR = Path("tmp")


@delayed
def generate_fold_obs_vs_pred(fold_dir: Path, xy: pd.DataFrame) -> pd.DataFrame:
    """Process a single fold of data."""
    predictor = TabularPredictor.load(str(fold_dir))
    pred = pd.DataFrame({"pred": predictor.predict(xy)})
    return pd.concat([xy[["x", "y", "obs"]], pred], axis=1)


def get_stats(
    cv_obs_vs_pred: pd.DataFrame,
    resolution: int | float,
    transform: str | None = None,
    wt_pearson: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Calculate statistics for a given DataFrame of observed and predicted values."""
    obs = cv_obs_vs_pred.obs.to_numpy()
    pred = cv_obs_vs_pred.pred.to_numpy()

    if transform:
        if transform == "log":
            # scale obs and pred to avoid log(0)
            scale = abs(min(obs.min(), pred.min()) - 1)
            obs = np.log(obs + scale)
            pred = np.log(pred + scale)
        elif transform == "power":
            obs = yeo_johnson_transform(obs, **kwargs)
            pred = yeo_johnson_transform(pred, **kwargs)
        else:
            raise ValueError(f"Invalid transform: {transform}")
        cv_obs_vs_pred = cv_obs_vs_pred.assign(obs=obs, pred=pred)

    r2 = metrics.r2_score(obs, pred)
    pearsonr = cv_obs_vs_pred[["obs", "pred"]].corr().iloc[0, 1]

    pearsonr_wt = None
    if wt_pearson:
        pearsonr_wt = weighted_pearson_r(
            cv_obs_vs_pred.set_index(["y", "x"]),
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


def load_x() -> pd.DataFrame:
    """Load X data for a given trait set."""
    tmp_x_path = TMP_DIR / "cv_stats" / "x.parquet"

    if tmp_x_path.exists():
        log.info("Found cached X data. Loading...")
        return pd.read_parquet(tmp_x_path)

    client, cluster = init_dask(dashboard_address=get_config().dask_dashboard)

    x_mask = dd.read_parquet(get_predict_mask_fn())
    x_imp = dd.read_parquet(get_predict_imputed_fn())

    xy = (
        dd.read_parquet(get_y_fn(), columns=["x", "y", "source"])
        .query("source == 's'")
        .drop(columns=["source"])
    )

    x_imp_trait = (
        dd.merge(x_imp, xy, how="inner", on=["x", "y"]).compute().set_index(["y", "x"])
    )
    mask_trait = (
        dd.merge(x_mask, xy, how="inner", on=["x", "y"]).compute().set_index(["y", "x"])
    )

    close_dask(client, cluster)

    x_trait_masked = x_imp_trait.mask(mask_trait)
    tmp_x_path.parent.mkdir(parents=True, exist_ok=True)
    x_trait_masked.to_parquet(tmp_x_path)
    return x_trait_masked


def load_y(trait_id: str) -> pd.DataFrame:
    """Load Y data for a given trait set."""
    y = (
        dd.read_parquet(get_y_fn(), columns=["x", "y", trait_id, "source"])
        .query("source == 's'")
        .merge(
            dd.read_parquet(get_cv_splits_dir() / f"{trait_id}.parquet"),
            how="inner",
            on=["x", "y"],
        )[["y", "x", trait_id, "fold"]]
    )

    return y.compute().set_index(["y", "x"])


def load_xy(x: pd.DataFrame, trait_id: str) -> pd.DataFrame:
    """Load X and Y data for a given trait set."""
    y = load_y(trait_id)
    return x.join(y, how="inner").reset_index().rename({trait_id: "obs"}, axis=1)


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument(
        "-r", "--recompute", action="store_true", help="Recompute stats."
    )
    parser.add_argument(
        "-p", "--persist", action="store_true", help="Persist temp. X data."
    )
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """
    Main function for generating spatial CV trait statistics.
    """
    models_dir = get_models_dir() / "debug" if args.debug else get_models_dir()
    trait_sets = ["splot", "splot_gbif", "gbif"]

    log.info("Loading X data...")
    x = load_x()
    for trait_dir in models_dir.iterdir():
        if not trait_dir.is_dir():
            continue

        trait_id = trait_dir.stem

        for trait_set in trait_sets:
            log.info("Processing trait: %s for %s", trait_id, trait_set)
            ts_dir = get_latest_run(trait_dir / cfg.train.arch) / trait_set
            if not ts_dir.exists():
                log.error("Skipping trait set: %s", trait_set)
                continue

            results_path = Path(ts_dir, cfg.train.eval_results)
            old_path = results_path.with_name(
                f"{results_path.stem}_old{results_path.suffix}"
            )
            if not results_path.exists():
                raise FileNotFoundError(f"Results file not found: {results_path}")

            if results_path.exists() and old_path.exists():
                if not args.recompute:
                    log.info("Found existing stats. Skipping...")
                    continue

                log.info("Found existing stats but recompute is True. Recomputing...")
                results_path.unlink()
                old_path.rename(results_path)

            log.info("Joining X and Y data...")
            trait_df = load_xy(x, trait_id)

            fold_dirs = [d for d in Path(ts_dir, "cv").iterdir() if d.is_dir()]
            fold_ids = [fold_dir.stem.split("_")[-1] for fold_dir in fold_dirs]

            log.info("Splitting folds...")
            fold_dfs = [
                trait_df.query(f"fold == {fold_id}")
                .drop(columns=["fold"])
                .reset_index(drop=True)
                for fold_id in fold_ids
            ]

            delayed_results = [
                generate_fold_obs_vs_pred(fold_dir, fold_df)
                for fold_dir, fold_df in zip(fold_dirs, fold_dfs)
            ]
            log.info("Computing delayed results...")
            coll = compute(*delayed_results)

            log.info("Concatenating results...")
            cv_obs_vs_pred = pd.concat(coll, ignore_index=True)

            log.info("Writing results to disk...")
            cv_obs_vs_pred_path = Path(ts_dir, "cv_obs_vs_pred.parquet")
            if cv_obs_vs_pred_path.exists():
                cv_obs_vs_pred_path.unlink()
            cv_obs_vs_pred.to_parquet(cv_obs_vs_pred_path)

            log.info("Calculating stats...")
            all_stats = pd.DataFrame()
            pearsonr_wt = cfg.crs == "EPSG:4326"

            # Back-transform if training data was log-transformed
            if cfg.trydb.interim.transform == "log":
                if "ln" in trait_id.split("_"):
                    log.info(
                        "Log-transformed trait detected. Back-transforming prior to "
                        "stats calculation..."
                    )
                    cv_obs_vs_pred = cv_obs_vs_pred.assign(
                        obs=np.expm1(cv_obs_vs_pred.obs),
                        pred=np.expm1(cv_obs_vs_pred.pred),
                    )
            # Back-transform if training data was power-transformed
            elif cfg.trydb.interim.transform == "power":
                with open(get_power_transformer_fn(cfg), "rb") as f:
                    pt = pickle.load(f)

                log.info("Inverse transforming Y data...")
                trait_num = get_trait_number_from_id(trait_id)
                feature_nums = np.array(
                    [get_trait_number_from_id(f) for f in pt.feature_names_in_]
                )
                lmbda = pt.lambdas_[np.where(feature_nums == trait_num)[0][0]]
                inv = yeo_johnson_inverse_transform(
                    cv_obs_vs_pred[["obs", "pred"]].to_numpy(), lmbda
                )
                cv_obs_vs_pred = cv_obs_vs_pred.assign(obs=inv[:, 0], pred=inv[:, 1])

            # Get the stats on the non-transformed data
            stats = get_stats(
                cv_obs_vs_pred, cfg.target_resolution, wt_pearson=pearsonr_wt
            )

            # Get the stats on the transformed data (yes, this is a little redundant)
            stats_tr = get_stats(
                cv_obs_vs_pred,
                cfg.target_resolution,
                transform=cfg.trydb.interim.transform,
                wt_pearson=pearsonr_wt,
            )

            all_stats = pd.concat(
                [
                    pd.DataFrame(stats, index=[0]).assign(transform="none"),
                    pd.DataFrame(stats_tr, index=[0]).assign(
                        transform=cfg.trydb.interim.transform
                    ),
                ],
                ignore_index=True,
            )

            log.info("Writing stats to disk...")
            results_path.rename(old_path)
            all_stats.to_csv(results_path, index=False)

    if not args.persist:
        log.info("Cleaning up temporary files...")
        shutil.rmtree(TMP_DIR)


if __name__ == "__main__":
    main()

"""Predict traits using best and most recent models."""

import argparse
import errno
import shutil
from pathlib import Path
from typing import Generator

import dask.dataframe as dd
import pandas as pd
from autogluon.tabular import TabularPredictor
from box import ConfigBox
from tqdm import tqdm

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, df_to_dd, init_dask
from src.utils.dataset_utils import (
    get_cov_dir,
    get_latest_run,
    get_models_dir,
    get_predict_dir,
    get_predict_imputed_fn,
    get_predict_mask_fn,
)
from src.utils.df_utils import grid_df_to_raster, pipe_log


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict traits using best and most recent models."
    )
    parser.add_argument(
        "--cov",
        action="store_true",
        help="Calculate Coefficient of Variation (instead of normal prediction)",
    )
    parser.add_argument(
        "-b",
        "--batches",
        type=int,
        default=24,
        help="Number of batches for prediction",
    )
    parser.add_argument(
        "-n",
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")

    parser.add_argument("-r", "--resume", action="store_true", help="Resume prediction")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    return parser.parse_args()


def predict_trait_ag_dask(
    data: dd.DataFrame,
    model_path: Path,
) -> pd.DataFrame:
    """Predict the trait using the given model in batches, optimized for Dask DataFrames,
    ensuring order is preserved."""
    log.info("Prediction with Dask...")
    return (
        data.map_partitions(predict_partition, predictor_path=model_path)
        .compute()
        .set_index(["y", "x"])
    )


def predict_trait_ag(data: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """Predict the trait using the given model in batches."""
    model = TabularPredictor.load(str(model_path))
    full_prediction = model.predict(data.drop(columns=["x", "y"]), as_pandas=True)

    # Concatenate xy DataFrame with predictions and set index
    result = pd.concat(
        [
            data[["x", "y"]].reset_index(drop=True),
            full_prediction.reset_index(  # pyright: ignore[reportAttributeAccessIssue]
                drop=True
            ),
        ],
        axis=1,
    )
    return result.set_index(["y", "x"])


def predict_partition(partition: pd.DataFrame, predictor_path: Path):
    """Predict on a single partition."""
    fold_predictor = TabularPredictor.load(str(predictor_path))

    predictions = fold_predictor.predict(
        partition.drop(columns=["x", "y"]), as_pandas=True
    )

    return pd.concat(
        [
            partition[["x", "y"]].reset_index(drop=True),
            predictions.reset_index(  # pyright: ignore[reportAttributeAccessIssue]
                drop=True
            ),
        ],
        axis=1,
    )


def predict_cov_dask(
    predict_data: dd.DataFrame, cv_dir: Path
) -> tuple[pd.DataFrame, Path]:
    """Calculate the Coefficient of Variation for the given model using parallel predictions."""
    cv_predictions = []
    tmp_dir = get_cov_dir(get_config()) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for fold_model_path in cv_dir.iterdir():
        cv_prediction_fn = Path(tmp_dir) / f"{fold_model_path.stem}.parquet"

        if not fold_model_path.is_dir():
            continue

        if cv_prediction_fn.exists():
            log.info("Skipping %s, already exists", cv_prediction_fn)
            cv_predictions.append(cv_prediction_fn)
            continue

        log.info("Predicting with %s...", fold_model_path.stem)
        predict_data.map_partitions(
            predict_partition, predictor_path=fold_model_path
        ).compute().set_index(["y", "x"]).to_parquet(cv_prediction_fn)

        cv_predictions.append(cv_prediction_fn)

    log.info("CV predictions complete. Re-loading...")
    dfs = [pd.read_parquet(f) for f in cv_predictions]

    log.info("Calculating CoV...")
    cov = (
        pd.concat(dfs, axis=1)
        .pipe(lambda _df: _df.std(axis=1) / _df.mean(axis=1))  # CoV calculation
        .rename("cov")
        .to_frame()
    )

    return cov, tmp_dir


def predict_dask(
    predict_data: dd.DataFrame, model_path: Path, cov: bool
) -> tuple[pd.DataFrame, Path | None]:
    """Predict the trait using the given model in batches, optimized for Dask DataFrames,
    ensuring order is preserved."""
    if cov:
        return predict_cov_dask(predict_data, model_path / "cv")
    return predict_trait_ag_dask(predict_data, model_path / "full_model"), None


def predict_traits_ag(
    predict_data: pd.DataFrame | dd.DataFrame,
    trait_model_dirs: list[Path] | Generator[Path, None, None],
    res: int | float,
    out_dir: str | Path,
    predict_cfg: ConfigBox,
    resume: bool = False,
    cov: bool = False,
) -> None:
    """Predict all traits that have been trained."""
    dask: bool = predict_cfg.batches > 1

    for trait_dir in (pbar := tqdm(list(trait_model_dirs))):
        if not trait_dir.is_dir():
            log.warning("Skipping %s, not a directory", trait_dir)
            continue

        trait: str = trait_dir.stem
        latest_run = get_latest_run(trait_dir / "autogluon")

        for trait_set_dir in latest_run.iterdir():
            if not trait_set_dir.is_dir() or trait_set_dir.stem == "gbif":
                continue

            pbar.set_description(f"{trait} -- {trait_set_dir.stem}")

            trait_set = trait_set_dir.stem
            out_fn = (
                Path(out_dir)
                / trait
                / trait_set
                / f"{trait}_{trait_set}_{'cov' if cov else 'predict'}.tif"
            )
            out_fn.parent.mkdir(parents=True, exist_ok=True)

            if resume and out_fn.exists():
                log.info("Skipping %s, already exists", out_fn)
                continue

            if dask:
                if isinstance(predict_data, pd.DataFrame):
                    predict_data = df_to_dd(
                        predict_data, npartitions=predict_cfg.batches
                    )

                client, cluster = init_dask(
                    dashboard_address=get_config().dask_dashboard,
                    n_workers=predict_cfg.n_workers,
                    threads_per_worker=1,
                )

                pred, tmp_dir = predict_dask(predict_data, trait_set_dir, cov)
                close_dask(client, cluster)
            else:
                log.info("Predicting traits for %s...", trait_set_dir)
                if isinstance(predict_data, dd.DataFrame):
                    predict_data = predict_data.compute()
                pred = predict_trait_ag(
                    pd.DataFrame(predict_data), trait_set_dir / "full_model"
                )
                tmp_dir = None

            log.info("Writing predictions to raster...")
            grid_df_to_raster(pred, res, out_fn)

            if tmp_dir is not None:
                shutil.rmtree(tmp_dir)


def load_predict(tmp_predict_fn: Path, batches: int = 1) -> pd.DataFrame | dd.DataFrame:
    """Load masked predict data from disk or mask imputed features."""
    log.info("Checking for existing masked predict data...")
    if not tmp_predict_fn.exists():
        log.info("No existing masked predict data found. Masking imputed features...")
        predict = (
            dd.read_parquet(get_predict_imputed_fn())
            .pipe(pipe_log, "Reading imputed predict features...")
            .compute()
            .reset_index(drop=True)
            .pipe(pipe_log, "Setting index to ['y', 'x']")
            .set_index(["y", "x"])
            .pipe(pipe_log, "Reading mask and masking imputed features...")
            .mask(pd.read_parquet(get_predict_mask_fn()).set_index(["y", "x"]))
            .reset_index()
        )

        predict.to_parquet(tmp_predict_fn, compression="zstd")

    else:
        log.info("Found existing masked predict data. Reading...")

    return (
        pd.read_parquet(tmp_predict_fn)
        if batches == 1
        else dd.read_parquet(tmp_predict_fn).repartition(npartitions=batches)
    )


def main(args: argparse.Namespace, cfg: ConfigBox = get_config()) -> None:
    """
    Predict the traits for the given model.
    """
    predict_cfg = cfg.predict[detect_system()]

    if not args.verbose:
        log.setLevel("WARNING")

    models_dir = get_models_dir(cfg)

    # E.g. ./data/processed/Shrub_Tree_Grass/001/predict
    out_dir = get_cov_dir(cfg) if args.cov else get_predict_dir(cfg)

    if args.debug:
        models_dir = models_dir / "debug"
        models_dir.mkdir(parents=True, exist_ok=True)
        out_dir = out_dir / "debug"

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_predict_fn = Path(out_dir / "predict.parquet")

    predict = load_predict(tmp_predict_fn, predict_cfg.batches)

    if cfg.train.arch == "autogluon":
        model_dirs = models_dir.glob("*")
        predict_traits_ag(
            predict_data=predict,
            trait_model_dirs=model_dirs,
            res=cfg.target_resolution,
            predict_cfg=predict_cfg,
            out_dir=out_dir,
            resume=args.resume,
            cov=args.cov,
        )
    else:
        raise NotImplementedError("Only Autogluon models are currently supported.")

    log.info("Cleaning up temporary files...")
    tmp_predict_fn.unlink()
    log.info("Done!")


if __name__ == "__main__":
    i = 0
    try:
        main(cli())
    except OSError as e:  # Except
        if e.errno == errno.EHOSTDOWN:  # Check for specific errno
            i += 1
            log.error("OSError: [Errno %s] Host is down: %s", e.errno, e.strerror)
            if i < 3:
                log.error("Retrying...")
                main(cli())
        else:
            raise

"""Predict traits using best and most recent models."""

import argparse
import tempfile
from pathlib import Path
from typing import Generator

import dask.dataframe as dd
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from box import ConfigBox
from tqdm import tqdm

from src.conf.conf import get_config
from src.conf.environment import log
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
        "-b", "--batches", type=int, default=1, help="Number of batches for prediction"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")

    parser.add_argument("-r", "--resume", action="store_true", help="Resume prediction")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    return parser.parse_args()


def predict_trait_ag_dask(
    data: dd.DataFrame,
    model: TabularPredictor,
    batches: int = 2,
) -> pd.DataFrame:
    """Predict the trait using the given model in batches, optimized for Dask DataFrames,
    ensuring order is preserved."""
    log.info("Repartitioning data into %d batches...", batches)
    data = data.repartition(npartitions=batches)

    predictions = []
    log.info("Predicting in batches...")
    for partition in data.to_delayed():
        batch = dd.from_delayed(partition)
        batch_pd = batch.compute()
        xy = batch_pd[["x", "y"]]
        batch_pd = batch_pd.drop(columns=["x", "y"])
        predictions.append(
            pd.concat(
                [
                    xy.reset_index(drop=True),
                    model.predict(batch_pd, as_pandas=True).reset_index(drop=True),
                ],
                axis=1,
            )
        )

    # Concatenate all batch predictions using Dask
    log.info("Combining batch predictions...")
    return pd.concat(predictions).reset_index(drop=True).set_index(["y", "x"])


def predict_trait_ag(data: pd.DataFrame, model: TabularPredictor) -> pd.DataFrame:
    """Predict the trait using the given model in batches."""
    full_prediction = model.predict(data.drop(columns=["x", "y"]), as_pandas=True)

    # Concatenate xy DataFrame with predictions and set index
    result = pd.concat(
        [
            TabularDataset(data[["x", "y"]].reset_index(drop=True)),
            full_prediction.reset_index(drop=True),
        ],
        axis=1,
    )
    return result.set_index(["y", "x"])


def predict_trait(
    predict_data: pd.DataFrame | dd.DataFrame,
    full_model: TabularPredictor,
    batches: int,
) -> pd.DataFrame:
    """Predict the trait using the given model."""
    if batches > 1:
        log.info("Batches > 1. Predicting in batches with Dask...")
        if isinstance(predict_data, pd.DataFrame):
            predict_data = dd.from_pandas(predict_data, npartitions=batches)

        pred = predict_trait_ag_dask(predict_data, full_model, batches)
    else:
        log.info("No batches detected. Loading full data and predicting...")
        pred = predict_trait_ag(
            predict_data,
            full_model,
        )

    return pred


def coefficient_of_variation(
    predict_data: dd.DataFrame, cv_dir: Path, batches: int = 2
) -> pd.DataFrame:
    """Calculate the Coefficient of Variation for the given model."""
    predict_data = predict_data.repartition(npartitions=batches)

    # xy = predict_data[["x", "y"]]
    # predict_data = predict_data.drop(columns=["x", "y"])

    with tempfile.TemporaryDirectory() as temp_dir:
        cv_predictions = []

        for fold_model in cv_dir.iterdir():
            if not fold_model.is_dir():
                continue

            log.info("Predicting with %s...", fold_model.stem)
            fold_predictor = TabularPredictor.load(str(fold_model))

            if batches > 1:
                batch_predictions = []
                # batch_size = len(predict_data) // batches + (len(predict_data) % batches > 0)

                # for i in range(batches):
                #     batch_data = predict_data.iloc[i * batch_size : (i + 1) * batch_size]
                #     batch_predictions.append(
                #         pd.DataFrame(
                #             fold_predictor.predict(batch_data),
                #             columns=[f"{fold_model.stem}"],
                #             index=batch_data.index,
                #         )
                #     )

                for partition in predict_data.to_delayed():
                    batch = dd.from_delayed(partition)
                    batch_pd = batch.compute()
                    xy = batch_pd[["x", "y"]]
                    batch_pd = batch_pd.drop(columns=["x", "y"])
                    batch_predictions.append(
                        pd.concat(
                            [
                                xy.reset_index(drop=True),
                                fold_predictor.predict(
                                    batch_pd, as_pandas=True
                                ).reset_index(drop=True),
                            ],
                            axis=1,
                        )
                    )

                fold_predictions = pd.concat(batch_predictions)
            else:
                raise NotImplementedError("Batches == 1 not implemented for CoV")

            prediction_file = Path(temp_dir) / f"{fold_model.stem}.parquet"
            fold_predictions.to_parquet(prediction_file, index=True)
            cv_predictions.append(prediction_file)

        log.info("CV predictions complete. Re-loading...")
        dfs = [pd.read_parquet(f) for f in cv_predictions]

        log.info("Calculating CoV...")
        cov = (
            pd.concat(dfs, axis=1)
            .pipe(lambda _df: _df.std(axis=1) / _df.mean(axis=1))  # CoV calculation
            .rename("cov")
            .pipe(lambda _df: pd.concat([xy, _df], axis=1))
            .set_index(["y", "x"])
        )

        return cov


def predict_traits_ag(
    predict_data: pd.DataFrame | dd.DataFrame,
    trait_model_dirs: list[Path] | Generator[Path, None, None],
    res: int | float,
    out_dir: str | Path,
    batches: int = 1,
    resume: bool = False,
    cov: bool = False,
) -> None:
    """Predict all traits that have been trained."""
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

            trait_set: str = trait_set_dir.stem
            out_fn: Path = (
                Path(out_dir)
                / trait
                / trait_set
                / f"{trait}_{trait_set}_{'cov' if cov else 'predict'}.tif"
            )
            out_fn.parent.mkdir(parents=True, exist_ok=True)

            if resume and out_fn.exists():
                log.info("Skipping %s, already exists", out_fn)
                continue

            if cov:
                log.info("Generating Coefficient of Variation for %s...", trait_set_dir)
                if batches == 1:
                    raise NotImplementedError("Batches == 1 not implemented for CoV")

                if isinstance(predict_data, pd.DataFrame):
                    predict_data = dd.from_pandas(predict_data, npartitions=batches)

                pred = coefficient_of_variation(
                    predict_data, trait_set_dir / "cv", batches
                )
            else:
                log.info("Predicting traits for %s...", trait_set_dir)
                full_model = TabularPredictor.load(str(trait_set_dir / "full_model"))
                pred = predict_trait(predict_data, full_model, batches)

            log.info("Writing predictions to raster...")
            grid_df_to_raster(pred, res, out_fn)


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
        predict = (
            pd.read_parquet(tmp_predict_fn)
            if batches == 1
            else dd.read_parquet(tmp_predict_fn)
        )

    return predict


def main(args: argparse.Namespace, cfg: ConfigBox = get_config()) -> None:
    """
    Predict the traits for the given model.
    """
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

    predict = load_predict(tmp_predict_fn, args.batches)

    if cfg.train.arch == "autogluon":
        model_dirs = models_dir.glob("*")
        predict_traits_ag(
            predict_data=predict,
            trait_model_dirs=model_dirs,
            res=cfg.target_resolution,
            out_dir=out_dir,
            batches=args.batches,
            resume=args.resume,
            cov=args.cov,
        )

    log.info("Cleaning up temporary files...")
    tmp_predict_fn.unlink()
    log.info("Done!")


if __name__ == "__main__":
    main(cli())

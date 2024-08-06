"""Predict traits using best and most recent models."""

import argparse
from pathlib import Path
from typing import Generator

import dask.dataframe as dd
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from box import ConfigBox
from dask.diagnostics import ProgressBar

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.autogluon_utils import get_best_model_ag
from src.utils.dataset_utils import get_models_dir, get_predict_fn
from src.utils.df_utils import grid_df_to_raster


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict traits using best and most recent models."
    )

    parser.add_argument(
        "-b", "--batches", type=int, default=1, help="Number of batches for prediction"
    )

    parser.add_argument("-r", "--resume", action="store_true", help="Resume prediction")
    return parser.parse_args()


def predict_trait_ag_dask(
    data: dd.DataFrame,
    model_path: str | Path,
    batches: int = 2,
) -> pd.DataFrame:
    """Predict the trait using the given model in batches, optimized for Dask DataFrames,
    ensuring order is preserved."""

    log.info("Loading model...")
    model = TabularPredictor.load(str(model_path))

    log.info("Repartitioning data into %d batches...", batches)
    data = data.repartition(npartitions=batches)

    predictions = []
    log.info("Predicting in batches...")
    with ProgressBar():
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


def predict_trait_ag(
    data: pd.DataFrame | TabularDataset,
    model_path: str | Path,
) -> pd.DataFrame:
    """Predict the trait using the given model in batches."""
    xy = data[["x", "y"]]  # Save the x and y columns
    data = data.drop(columns=["x", "y"])

    model = TabularPredictor.load(str(model_path))
    full_prediction = model.predict(data, as_pandas=True)

    # Concatenate xy DataFrame with predictions and set index
    result = pd.concat(
        [xy.reset_index(drop=True), full_prediction.reset_index(drop=True)], axis=1
    )
    return result.set_index(["y", "x"])


def predict_traits_ag(
    data_path: str | Path,
    trait_model_dirs: list[Path] | Generator[Path, None, None],
    res: int | float,
    out_dir: str | Path,
    batches: int = 1,
    resume: bool = False,
) -> None:
    """Predict all traits that have been trained."""
    for trait_models in trait_model_dirs:
        if not trait_models.is_dir():
            log.warning("Skipping %s, not a directory", trait_models)
            continue

        out_fn: Path = Path(out_dir) / f"{trait_models.stem}.tif"

        if resume and out_fn.exists():
            log.info("Skipping %s, already exists", out_fn)
            continue

        log.info("Predicting traits for %s...", trait_models)
        best_model_path = get_best_model_ag(trait_models)

        if batches > 1:
            log.info("Batches > 1. Predicting in batches with Dask...")
            pred = predict_trait_ag_dask(
                dd.read_parquet(data_path), best_model_path, batches
            )
        else:
            log.info("No batches detected. Loading full data and predicting...")
            pred = predict_trait_ag(
                dd.read_parquet(data_path).compute().reset_index(drop=True),
                best_model_path,
            )

        log.info("Writing predictions to raster...")
        grid_df_to_raster(pred, res, out_fn)


def main(args: argparse.Namespace, cfg: ConfigBox = get_config()) -> None:
    """
    Predict the traits for the given model.
    """

    predict_fn: Path = get_predict_fn(cfg)

    models_dir: Path = get_models_dir(cfg)

    # E.g. ./data/processed/Shrub_Tree_Grass/001/splot_gbif/predict
    out_dir = (
        Path(cfg.processed.dir)
        / cfg.PFT
        / cfg.model_res
        / cfg.datasets.Y.use
        / cfg.processed.predict_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.train.arch == "autogluon":
        model_dirs = models_dir.glob("*")
        predict_traits_ag(
            predict_fn,
            model_dirs,
            cfg.target_resolution,
            out_dir,
            args.batches,
            args.resume,
        )


if __name__ == "__main__":
    main(cli())

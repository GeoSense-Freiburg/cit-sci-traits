"""Predict traits using best and most recent models."""

import argparse
from pathlib import Path
from typing import Generator

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from box import ConfigBox
from tqdm import trange

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.autogluon_utils import get_best_model_ag
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


def predict_trait_ag(
    data: pd.DataFrame | TabularDataset,
    xy: pd.DataFrame,
    model_path: str | Path,
    batches: int = 1,
) -> pd.DataFrame:
    """Predict the trait using the given model in batches."""
    model = TabularPredictor.load(str(model_path))

    if batches > 1:
        # Calculate batch size
        batch_size = len(data) // batches + (len(data) % batches > 0)

        # Initialize an empty list to store batch predictions
        predictions = []

        # Predict in batches
        log.info("Predicting in batches...")
        for i in trange(0, len(data), batch_size):
            batch = data.iloc[i : i + batch_size]
            predictions.append(model.predict(batch, as_pandas=True))

        # Concatenate all batch predictions
        full_prediction = pd.concat(predictions)
    else:
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
    log.info("Loading predict data...")
    data = pd.read_parquet(data_path)
    # Add super small noise to vodca columns to avoid AutoGluon's super annoying type
    # coercion
    for col in data.columns:
        if col.startswith("vodca"):
            data[col] += 1e-10
    data = TabularDataset(data)

    xy = data[["x", "y"]]  # Save the x and y columns
    data = data.drop(columns=["x", "y"])

    for trait_models in trait_model_dirs:
        if not trait_models.is_dir():
            log.warning("Skipping %s, not a directory", trait_models)
            continue

        out_fn: Path = Path(out_dir) / f"{trait_models.stem}.tif"

        if resume and out_fn.exists():
            log.info("Skipping %s, already exists", trait_models)
            continue

        log.info("Predicting traits for %s...", trait_models)
        best_model_path = get_best_model_ag(trait_models)
        pred = predict_trait_ag(data, xy, best_model_path, batches)

        log.info("Writing predictions to raster...")
        grid_df_to_raster(pred, res, out_fn)


def main(args: argparse.Namespace, cfg: ConfigBox = get_config()) -> None:
    """
    Predict the traits for the given model.
    """

    predict_fn: Path = (
        Path(cfg.train.dir)
        / cfg.eo_data.predict.dir
        / cfg.model_res
        / cfg.eo_data.predict.filename
    )

    models_dir: Path = (
        Path(cfg.models.dir)
        / cfg.PFT
        / cfg.model_res
        / cfg.datasets.Y.use
        / cfg.train.arch
    )

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

"""Predict traits using best and most recent models."""

from pathlib import Path
from typing import Generator
from box import ConfigBox
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.df_utils import grid_df_to_raster


def get_best_model_ag(models_dir: Path) -> Path:
    """Find the best model in the specified directory."""
    quality_levels = ["best", "high", "medium", "fastest"]
    # Initialize the variables to store the best model and its timestamp
    best_model = None

    # Loop over the quality levels in descending order
    for quality in quality_levels:
        # Get the directories for the current quality level
        models = sorted(
            [
                d
                for d in models_dir.iterdir()
                if d.is_dir() and d.name.startswith(quality)
            ],
            reverse=True,
        )

        if not models:
            continue

        best_model = models[0]
        break

    if best_model is None:
        raise ValueError("No models found in the specified directory")

    return best_model


def predict_trait_ag(
    data: pd.DataFrame | TabularDataset, xy: pd.DataFrame, model_path: str | Path
) -> pd.DataFrame:
    """Predict the trait using the given model."""
    model = TabularPredictor.load(str(model_path))
    pred = model.predict(data, as_pandas=True)
    return pd.concat([xy, pred], axis=1).set_index(["y", "x"])  # type: ignore


def predict_traits_ag(
    data_path: str | Path,
    trait_model_dirs: list[Path] | Generator[Path, None, None],
    res: int | float,
    out_dir: str | Path,
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
        log.info("Predicting traits for %s...", trait_models)
        best_model_path = get_best_model_ag(trait_models)
        pred = predict_trait_ag(data, xy, best_model_path)

        log.info("Writing predictions to raster...")
        grid_df_to_raster(pred, res, Path(out_dir) / f"{trait_models.stem}.tif")


def main(cfg: ConfigBox = get_config()) -> None:
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
        )


if __name__ == "__main__":
    main()

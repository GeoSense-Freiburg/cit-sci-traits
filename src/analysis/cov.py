"""Calculate coefficient of variation for predictions from AutoGluon models."""

import argparse
import tempfile
from pathlib import Path

import dask.dataframe as dd
import joblib
import pandas as pd
from autogluon.tabular import TabularPredictor
from box import ConfigBox
from tqdm import trange

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.autogluon_utils import get_best_model_ag
from src.utils.dataset_utils import (
    get_models_dir,
    get_predict_imputed_fn,
    get_predict_mask_fn,
)
from src.utils.df_utils import grid_df_to_raster


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batches", type=int, default=1, help="Number of batches"
    )
    parser.add_argument("-r", "--resume", action="store_true", help="Resume prediction")
    return parser.parse_args()


def calculate_cov_ag(
    data: pd.DataFrame,
    base_model_dir: str | Path,
    batches: int = 1,
) -> pd.DataFrame:
    """Calculate coefficient of variation for predictions from AutoGluon models using Dask."""
    xy = data[["x", "y"]]
    data = data.drop(columns=["x", "y"])

    with tempfile.TemporaryDirectory() as temp_dir:
        prediction_files = []

        for fold_model in Path(base_model_dir).iterdir():
            if not fold_model.stem.startswith("S1") or not fold_model.is_dir():
                continue

            log.info(f"Predicting with {fold_model.stem}")
            fold_predictor = joblib.load(str(fold_model / "model.pkl"))

            if batches > 1:
                batch_predictions = []
                batch_size = len(data) // batches + (len(data) % batches > 0)

                for i in trange(batches):
                    batch_data = data.iloc[i * batch_size : (i + 1) * batch_size]
                    batch_predictions.append(
                        pd.DataFrame(
                            fold_predictor.predict(batch_data),
                            columns=[f"{fold_model.stem}"],
                            index=batch_data.index,
                        )
                    )

                fold_predictions = pd.concat(batch_predictions)
            else:
                fold_predictions = pd.DataFrame(
                    fold_predictor.predict(data),
                    columns=[f"{fold_model.stem}"],
                    index=data.index,
                )

            prediction_file = Path(temp_dir) / f"{fold_model.stem}.parquet"
            fold_predictions.to_parquet(prediction_file, index=True)
            prediction_files.append(prediction_file)

        log.info("Loading fold predictions...")
        dfs = [pd.read_parquet(f) for f in prediction_files]

        log.info("Calculating CoV...")
        cov = (
            pd.concat(dfs, axis=1)
            .pipe(lambda _df: _df.std(axis=1) / _df.mean(axis=1))  # CoV calculation
            .rename("cov")
            .pipe(lambda _df: pd.concat([xy, _df], axis=1))
            .set_index(["y", "x"])
        )

        return cov


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Main function"""
    log.info("Loading and masking predict data...")
    pred_imputed = (
        dd.read_parquet(get_predict_imputed_fn()).compute().set_index(["y", "x"])
    )
    pred_mask = dd.read_parquet(get_predict_mask_fn()).compute().set_index(["y", "x"])
    pred_data = pred_imputed.mask(pred_mask)

    models_dir = get_models_dir()

    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        out_fn: Path = Path(out_dir) / f"{model_dir.stem}_cov.tif"

        if args.resume and out_fn.exists():
            log.info("Skipping %s, already exists", model_dir)
            continue

        best_predictor = TabularPredictor.load(str(get_best_model_ag(model_dir)))

        # Get the best base model, excluding NN models as they don't seem to work the
        # same way as bagged models...
        log.info("Getting the best base model for %s...", model_dir.stem)
        best_base_model = (
            best_predictor.leaderboard(refit_full=False)
            .pipe(lambda df: df[df["stack_level"] == 1])
            .pipe(lambda df: df[~df["model"].str.contains("Neural")])
            .pipe(lambda df: df.loc[df["score_val"].idxmax()])
            .model
        )

        cv_models_dir = Path(best_predictor.path, "models", str(best_base_model))

        log.info(
            "Calculating coefficient of variation for %s using %s...",
            model_dir.stem,
            best_base_model,
        )
        cov = calculate_cov_ag(data, cv_models_dir, args.batches)

        log.info("Writing predictions to raster...")
        grid_df_to_raster(cov, cfg.target_resolution, out_fn)


if __name__ == "__main__":
    main()

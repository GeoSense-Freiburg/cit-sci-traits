"""Calculate coefficient of variation for predictions from AutoGluon models."""

import argparse
from pathlib import Path

from box import ConfigBox
import joblib
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from tqdm import trange

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.autogluon_utils import get_best_model_ag
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
    data: pd.DataFrame | TabularDataset,
    xy: pd.DataFrame,
    base_model_dir: str | Path,
    batches: int = 1,
) -> pd.DataFrame:
    """Calculate coefficient of variation for predictions from AutoGluon models."""
    fold_predictions = []

    for fold_model in Path(base_model_dir).iterdir():
        if not fold_model.stem.startswith("S1") or not fold_model.is_dir():
            continue

        log.info("Predicting with %s", fold_model.stem)
        fold_predictor = joblib.load(str(fold_model / "model.pkl"))

        if batches > 1:
            batch_size = len(data) // batches + (len(data) % batches > 0)
            batch_predictions = []

            for i in trange(batches):
                batch_data = data.iloc[i * batch_size : (i + 1) * batch_size]
                batch_predictions.append(
                    pd.DataFrame(
                        fold_predictor.predict(batch_data),
                        columns=[f"{fold_model.stem}"],
                    )
                )

            batch_predictions = pd.concat(batch_predictions, axis=0, ignore_index=True)
            fold_predictions.append(batch_predictions)

        else:
            fold_predictions.append(
                pd.DataFrame(
                    fold_predictor.predict(data), columns=[f"{fold_model.stem}"]
                )
            )

    all_predictions = pd.concat(fold_predictions, axis=1)
    cov = all_predictions.std(axis=1) / all_predictions.mean(axis=1)
    cov.name = "cov"

    # Concatenate xy DataFrame with predictions and set index
    result = pd.concat([xy.reset_index(drop=True), cov.reset_index(drop=True)], axis=1)
    return result.set_index(["y", "x"])


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Main function"""
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
        / cfg.processed.cov_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading predict data...")
    data = pd.read_parquet(predict_fn)
    xy = data[["x", "y"]]
    data = data.drop(columns=["x", "y"])

    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        out_fn: Path = Path(out_dir) / f"{model_dir.stem}_cov.tif"

        if args.resume and out_fn.exists():
            log.info("Skipping %s, already exists", model_dir)
            continue

        best_predictor = TabularPredictor.load(str(get_best_model_ag(model_dir)))

        best_base_model = (
            best_predictor.leaderboard(refit_full=False)
            .pipe(lambda df: df[df["stack_level"] == 1])
            .pipe(lambda df: df.loc[df["score_val"].idxmax()])
            .model
        )

        cv_models_dir = Path(best_predictor.path, "models", str(best_base_model))

        log.info(
            "Calculating coefficient of variation for %s using %s...",
            model_dir.stem,
            best_base_model,
        )
        cov = calculate_cov_ag(data, xy, cv_models_dir, args.batches)

        log.info("Writing predictions to raster...")
        grid_df_to_raster(cov, cfg.target_resolution, out_fn)


if __name__ == "__main__":
    main()

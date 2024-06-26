"""Train a set of AutoGluon models using the given configuration."""

import datetime
import logging
from pathlib import Path
import pickle

import dask.dataframe as dd
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import get_models_dir, get_train_dir
from src.utils.log_utils import get_loggers_starting_with, setup_file_logger


def evaluate_model(
    predictor: TabularPredictor,
    y_true: pd.Series,
    y_pred: pd.Series,
    split: pd.Series,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Evaluate the model using the given evaluation metrics and save the results to a CSV."""
    cv_pred = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "split": split,
        }
    )

    # Group by split and calculate the evaluation metrics
    cv_eval = pd.DataFrame(
        cv_pred.groupby("split")
        .apply(
            lambda x: predictor.evaluate_predictions(  # pylint: disable=cell-var-from-loop
                y_true=x["y_true"], y_pred=x["y_pred"]
            )
        )
        .to_list()
    )

    # Calculate normalized RMSE for each group
    y_range = y_true.max() - y_true.min()
    cv_eval["norm_root_mean_squared_error"] = (
        cv_eval["root_mean_squared_error"] / y_range
    )

    # Convert the evaluation results to a DataFrame and calculate mean and std
    results = cv_eval.apply(pd.Series).agg(["mean", "std"])

    if out_path is not None:
        log.info("Saving evaluation results to %s...", out_path)
        results.to_csv(out_path, index=False)

    return results


def train_models(
    cfg: ConfigBox = get_config(), sample: float = 1.0, debug: bool = False
) -> None:
    """Train a set of AutoGluon models for each  using the given configuration."""
    train_dir = get_train_dir(cfg)
    model_dir = get_models_dir(cfg) / "debug" if debug else get_models_dir(cfg)
    model_dir.mkdir(parents=True, exist_ok=True)

    file_logger = setup_file_logger(
        "train.autogluon", model_dir / "log.txt", level=logging.ERROR
    )

    dask_loggers = get_loggers_starting_with("distributed")
    for logger_name in dask_loggers:
        logging.getLogger(logger_name).setLevel("WARNING")

    log.info("Loading data...")
    feats = dd.read_parquet(train_dir / cfg.train.features).drop(columns=["x", "y"])
    y_cols = feats.columns[feats.columns.str.startswith("X")].to_list()
    x_cols = feats.columns[~feats.columns.str.startswith("X")].to_list()

    # Select only the traits that correspond with the numbers in cfg.datasets.Y.traits
    # y_cols are in format "X{trait_number}_{trait_stat}"
    y_cols = [
        y_col
        for y_col in y_cols
        if int(y_col.split("_")[0][1:]) in cfg.datasets.Y.traits
    ]

    for y_col in y_cols:
        # Select all X_cols and first entry of Y_cols from feats
        xy = feats[x_cols + [y_col]].compute().reset_index(drop=True)

        log.info("Training model for %s...", y_col)
        log.info("Assigning CV splits...")
        # Load the CV splits
        with open(train_dir / cfg.train.cv_splits.dir / f"{y_col}.pkl", "rb") as f:
            cv_splits = pickle.load(f)

        # Each split is a tuple of (train_idx, valid_idx). Assign the split number to each set
        # of valid_idx in Xy
        for i, (_, valid_idx) in enumerate(cv_splits):
            xy.loc[valid_idx, "split"] = i

        log.info("Training model...")
        if sample < 1.0:
            xy = xy.sample(frac=sample, random_state=cfg.random_seed)

        # Choose a random split ID from the range of split values
        np.random.seed(cfg.random_seed)
        val_split_id = np.random.choice(xy["split"].unique())

        # Use the random split ID to split the data into train and validation sets
        train = TabularDataset(xy[xy["split"] != val_split_id])
        val = TabularDataset(xy[xy["split"] == val_split_id])

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / y_col / f"{cfg.autogluon.presets}_{now}"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            predictor = TabularPredictor(
                label=y_col, groups="split", path=model_path
            ).fit(
                train,
                included_model_types=cfg.autogluon.included_model_types,
                num_cpus=cfg.autogluon.num_cpus,
                num_gpus=cfg.autogluon.num_gpus,
                presets=cfg.autogluon.presets,
                time_limit=cfg.autogluon.time_limit,
                save_bag_folds=cfg.autogluon.save_bag_folds,
                refit_full=cfg.autogluon.refit_full,
                set_best_to_refit_full=cfg.autogluon.set_best_to_refit_full,
            )

            if cfg.autogluon.feature_importance:
                log.info("Calculating feature importance...")
                feature_importance = predictor.feature_importance(
                    val,
                    time_limit=cfg.autogluon.FI_time_limit,
                    num_shuffle_sets=cfg.autogluon.FI_num_shuffle_sets,
                )
                feature_importance.to_csv(model_path / cfg.train.feature_importance)

            log.info("Evaluating model...")
            _ = evaluate_model(
                predictor,
                train[y_col],
                predictor.predict_oof(train_data=train),
                train["split"],
                model_path / cfg.train.eval_results,
            )

            log.info("Refitting model on full data...")
            predictor.refit_full("best", train_data_extra=val)

            # Clean up the model directory
            predictor.save_space(remove_fit_stack=False)

        except ValueError as e:
            file_logger.error("Error training model: %s", e)
            continue

    log.info("Done! \U00002705")

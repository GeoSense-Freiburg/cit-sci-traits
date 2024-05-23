"""Train a set of AutoGluon models using the given configuration."""

import logging
from pathlib import Path
import pickle
import datetime

from box import ConfigBox

from autogluon.tabular import TabularDataset, TabularPredictor
import dask.dataframe as dd

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.log_utils import get_loggers_starting_with, setup_file_logger


def train(cfg: ConfigBox = get_config(), sample: float = 1.0) -> None:
    """Train a set of AutoGluon models for each  using the given configuration."""
    train_dir = Path(cfg.train.dir) / cfg.PFT / cfg.model_res / cfg.datasets.Y.use
    model_dir = (
        Path(cfg.models.dir)
        / cfg.PFT
        / cfg.model_res
        / cfg.datasets.Y.use
        / cfg.train.arch
    )
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

        train_idx = xy.sample(frac=0.9, random_state=cfg.random_seed).index
        test_idx = xy.index.difference(train_idx)
        train_data = TabularDataset(xy.loc[train_idx])
        test_data = TabularDataset(xy.loc[test_idx].drop(columns=["split"]))

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = model_dir / y_col / f"{cfg.autogluon.quality}_{now}"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            predictor = TabularPredictor(
                label=y_col, groups="split", path=model_path
            ).fit(
                train_data,
                num_bag_folds=cfg.train.cv_splits.n_splits,
                excluded_model_types=cfg.autogluon.exclude_models,
                num_cpus=90,  # pyright: ignore[reportArgumentType]
                presets=f"{cfg.autogluon.quality}_quality",
                time_limit=cfg.autogluon.time_limit,
            )

            log.info("Evaluating model...")
            evaluation = predictor.evaluate(test_data)
            # Save the evaluation results
            with open(model_path / "evaluation_results.pkl", "wb") as f:
                pickle.dump(evaluation, f)
        except ValueError as e:
            file_logger.error("Error training model: %s", e)
            continue

    log.info("Done! \U00002705")

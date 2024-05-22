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
from src.utils.log_utils import get_loggers_starting_with


def train(cfg: ConfigBox = get_config(), sample: float = 1.0) -> None:
    """Train a set of AutoGluon models for each  using the given configuration."""
    train_dir = Path(cfg.train.dir) / cfg.PFT / cfg.model_res / cfg.datasets.Y.use
    dask_loggers = get_loggers_starting_with("distributed")
    for logger_name in dask_loggers:
        logging.getLogger(logger_name).setLevel("WARNING")

    log.info("Loading data...")
    feats = dd.read_parquet(train_dir / cfg.train.features).drop(columns=["x", "y"])
    y_cols = feats.columns[feats.columns.str.startswith("X")].to_list()
    x_cols = feats.columns[~feats.columns.str.startswith("X")].to_list()

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
        model_dir = (
            Path(cfg.models.dir)
            / cfg.PFT
            / cfg.model_res
            / cfg.datasets.Y.use
            / cfg.train.arch
            / y_col
            / f"{cfg.autogluon.quality}_{now}"
        )
        model_dir.parent.mkdir(parents=True, exist_ok=True)

        predictor = TabularPredictor(label=y_col, groups="split", path=model_dir).fit(
            train_data,
            num_bag_folds=cfg.train.cv_splits.n_splits,
            excluded_model_types=["KNN"],
            num_cpus=90,  # pyright: ignore[reportArgumentType]
            presets=f"{cfg.autogluon.quality}_quality",
        )

        log.info("Evaluating model...")
        evaluation = predictor.evaluate(test_data)
        # Save the evaluation results
        with open(model_dir / "evaluation_results.pkl", "wb") as f:
            pickle.dump(evaluation, f)

    log.info("Done! \U00002705")

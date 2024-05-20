"""Split the data into train and test sets using spatial k-fold cross-validation."""

import logging
import pickle
from pathlib import Path
from typing import Sequence

import dask.bag as db
import dask.dataframe as dd
import numpy as np
import numpy.typing as npt
import pandas as pd
from box import ConfigBox
from dask import compute, delayed
from dask.distributed import Client
from scipy.stats import ks_2samp

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.log_utils import get_loggers_starting_with
from src.utils.spatial_utils import acr_to_h3_res, assign_hexagons


def calculate_kg_p_value(
    df: pd.DataFrame, data_col: str, fold_i: int, fold_j: int
) -> float:
    """
    Calculate the p-value using the Kolmogorov-Smirnov test for two folds in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        data_col (str): The column name of the data to compare.
        fold_i (int): The index of the first fold.
        fold_j (int): The index of the second fold.

    Returns:
        float: The p-value calculated using the Kolmogorov-Smirnov test.
    """
    fold_i_values = df[df["fold"] == fold_i][data_col]
    fold_j_values = df[df["fold"] == fold_j][data_col]
    _, p_value = ks_2samp(fold_i_values, fold_j_values)
    return p_value


def calculate_similarity_kg(folds: Sequence, df: pd.DataFrame, data_col: str) -> float:
    """
    Calculate the similarity between folds using the Kolmogorov-Smirnov test.

    Parameters:
    - folds (Sequence): A sequence of folds.
    - df (pd.DataFrame): The DataFrame containing the data.
    - data_col (str): The name of the column containing the data.

    Returns:
    - float: The similarity between the folds based on the Kolmogorov-Smirnov test.
    """

    # Use Dask's delayed function to parallelize the pairwise comparisons
    p_values = [
        delayed(calculate_kg_p_value)(df, data_col, folds[i], folds[j])
        for i in range(len(folds))
        for j in range(i + 1, len(folds))
    ]

    # Compute the p-values in parallel
    p_values = compute(*p_values)

    similarity = np.min(p_values)
    return similarity


def assign_folds_iteration(args):
    """
    Assigns folds to the hexagons in the given dataframe based on the number of folds
    specified.

    Args:
        args (tuple): A tuple containing the following elements:
            - df (pandas.DataFrame): The dataframe containing the hexagons.
            - n_folds (int): The number of folds to create.
            - data_col (str): The name of the column in the dataframe containing the data.
            - hexagons (numpy.ndarray): An array of hexagons.

    Returns:
        tuple: A tuple containing the following elements:
            - similarity (float): The similarity value calculated based on the assigned
                folds.
            - folds (pandas.Series): A copy of the "fold" column in the dataframe.

    """
    df, n_folds, data_col, hexagons = args
    np.random.shuffle(hexagons)
    folds = np.array_split(hexagons, n_folds)
    for j, fold in enumerate(folds):
        df.loc[df["hex_id"].isin(fold), "fold"] = j

    similarity = calculate_similarity_kg(range(n_folds), df, data_col)
    return similarity, df["fold"].copy()


def assign_folds(
    df: pd.DataFrame, n_folds: int, n_iterations: int, data_col: str
) -> pd.DataFrame:
    """
    Assigns folds to the given DataFrame based on similarity scores.

    Args:
        df (pd.DataFrame): The input DataFrame.
        n_folds (int): The number of folds to assign.
        n_iterations (int): The number of iterations to perform.
        data_col (str): The column name containing the data.

    Returns:
        pd.DataFrame: The input DataFrame with an additional 'fold' column.

    """
    hexagons = df["hex_id"].unique()
    best_similarity = None
    best_assignment = pd.Series(dtype=int)

    results = (
        db.from_sequence(
            [(df, n_folds, data_col, hexagons) for _ in range(n_iterations)]
        )
        .map(assign_folds_iteration)
        .compute()
    )

    for similarity, assignment in results:
        if best_similarity is None or similarity > best_similarity:
            best_similarity = similarity
            best_assignment = assignment

    log.info("Best similarity: %s", best_similarity)
    df["fold"] = best_assignment.astype(int)
    return df


def get_splits(
    df: pd.DataFrame,
) -> list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """
    Generate train-test splits based on the 'fold' column in the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data and the 'fold' column.

    Returns:
        list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]: A list of tuples,
            where each tuple contains the train and test indices for a fold.
    """
    splits = []
    folds = df["fold"].unique()
    for fold in folds:
        train = df[df["fold"] != fold].index.to_numpy()
        test = df[df["fold"] == fold].index.to_numpy()
        splits.append((train, test))
    return splits


def main(cfg: ConfigBox = get_config()) -> None:
    """Main function to generate spatial k-fold cross-validation splits."""
    train_dir = Path(cfg.train.dir) / cfg.PFT / cfg.model_res / cfg.datasets.Y.use
    ranges = pd.read_parquet(
        train_dir / cfg.train.spatial_autocorr,
        columns=["trait", cfg.train.cv_splits.range_stat],
    )

    feat_cols = dd.read_parquet(train_dir / cfg.train.features).columns

    # Only select columns starting with "X"
    feat_cols = [col for col in feat_cols if col.startswith("X")]
    feats = dd.read_parquet(
        train_dir / cfg.train.features, columns=[["x", "y"] + feat_cols]
    ).repartition(npartitions=100)

    for trait in feat_cols:
        log.info("Processing trait: %s", trait)

        with Client(dashboard_address=cfg.dask_dashboard):
            # Ensure dask loggers don't interfere with the main logger
            dask_loggers = get_loggers_starting_with("distributed")
            for logger in dask_loggers:
                logging.getLogger(logger).setLevel(logging.WARNING)

            trait_range = ranges[ranges["trait"] == trait][
                cfg.train.cv_splits.range_stat
            ]
            h3_res = acr_to_h3_res(trait_range)
            df = assign_hexagons(feats[["x", "y", trait]], h3_res, dask=True)
            df = assign_folds(
                df, cfg.train.cv_splits.n_splits, cfg.train.cv_splits.n_sims, trait
            )

        splits = get_splits(df)

        # Save the splits
        splits_dir = train_dir / cfg.train.cv_splits.dir
        splits_dir.mkdir(parents=True, exist_ok=True)

        with open(splits_dir / f"{trait}.pkl", "wb") as f:
            pickle.dump(splits, f)

    log.info("Done!")

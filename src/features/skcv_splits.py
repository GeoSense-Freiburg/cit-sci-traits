"""Split the data into train and test sets using spatial k-fold cross-validation."""

import argparse
import logging
import warnings
from typing import Sequence

import dask.dataframe as dd
import numpy as np
import numpy.typing as npt
import pandas as pd
from box import ConfigBox
from dask import compute, delayed
from scipy.stats import ks_2samp

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import get_autocorr_ranges_fn, get_cv_splits_dir, get_y_fn
from src.utils.df_utils import reproject_xy_to_geo
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
    folds_df = df[df["fold"].isin([fold_i, fold_j])]
    folds_values = folds_df[data_col]
    mask = folds_df["fold"] == fold_i
    fold_i_values = folds_values[mask]
    fold_j_values = folds_values[~mask]
    _, p_value = ks_2samp(fold_i_values, fold_j_values)
    return p_value  # pyright: ignore[reportReturnType]


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

    # Calculate the pairwise comparisons
    p_values = [
        calculate_kg_p_value(df, data_col, folds[i], folds[j])
        for i in range(len(folds))
        for j in range(i + 1, len(folds))
    ]

    # Return the mean p-value as the similarity score
    return float(np.mean(p_values))


def assign_folds_iteration(
    df: pd.DataFrame, n_folds: int, data_col: str, hexagons: npt.NDArray
) -> tuple[float, pd.Series]:
    """
    Assigns folds to the hexagons in the given dataframe based on the number of folds
    specified.

    Parameters:
    - df: The input dataframe containing the hexagon data.
    - n_folds: The number of folds to assign.
    - data_col: The column name in the dataframe containing the data.
    - hexagons: The array of hexagons to assign folds to.

    Returns:
    - A tuple containing the similarity score and a copy of the fold assignments.
    """
    np.random.shuffle(hexagons)
    folds = np.array_split(hexagons, n_folds)
    hexagon_to_fold = {hexagon: i for i, fold in enumerate(folds) for hexagon in fold}
    df["fold"] = df["hex_id"].map(hexagon_to_fold)

    similarity = calculate_similarity_kg(range(n_folds), df, data_col)
    return similarity, df["fold"].copy()


def assign_folds(
    df: pd.DataFrame, n_folds: int, n_iterations: int, data_col: str
) -> pd.DataFrame:
    """
    Assigns folds to the given DataFrame based on similarity scores.

    Args:
        df (pd.DataFrame): The DataFrame to assign folds to.
        n_folds (int): The number of folds to assign.
        n_iterations (int): The number of iterations to perform.
        data_col (str): The column name in the DataFrame containing the data.

    Returns:
        pd.DataFrame: The DataFrame with the folds assigned.

    """
    hexagons = df["hex_id"].unique()

    results = compute(
        *[
            delayed(assign_folds_iteration)(df, n_folds, data_col, hexagons)
            for _ in range(n_iterations)
        ]
    )

    def _compute_best_similarity(
        assignments: list[tuple[float, pd.Series]]
    ) -> tuple[pd.Series, float]:
        best_similarity = None
        best_assignment = pd.Series(dtype=int)

        for similarity, assignment in assignments:
            log.info("Similarity: %e. Current best: %e", similarity, best_similarity)
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_assignment = assignment

        if best_similarity is None:
            raise ValueError("No best similarity found.")

        return best_assignment, best_similarity

    best_assignment, best_similarity = _compute_best_similarity(results)
    log.info("Best similarity: %e", best_similarity)
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


def _assign_trait_splits(
    traits_df: dd.DataFrame,
    trait_col: str,
    ranges: pd.DataFrame,
    overwrite: bool,
    cfg: ConfigBox,
) -> None:
    splits_dir = get_cv_splits_dir()
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_fn = splits_dir / f"{trait_col}.parquet"

    if splits_fn.exists() and not overwrite:
        log.info("Splits for trait %s already exist. Skipping...", trait_col)
        return

    log.info("Processing trait: %s", trait_col)
    # Ensure dask loggers don't interfere with the main logger
    dask_loggers = get_loggers_starting_with("distributed")
    for logger in dask_loggers:
        logging.getLogger(logger).setLevel(logging.WARNING)

    trait_df = traits_df[[trait_col, "x", "y"]].dropna()

    trait_range = ranges[ranges["trait"] == trait_col][
        cfg.train.cv_splits.range_stat
    ].values[0]

    if cfg.crs == "EPSG:4326":
        trait_range_deg = trait_range / 111320
        if trait_range_deg <= cfg.target_resolution:
            log.warning(
                "Trait range of %.2f m is less than or equal to the existing map"
                "resolution of %.2f m. "
                "Using the map resolution for hexagon assignment...",
                trait_range,
                cfg.target_resolution,
            )
            trait_range = cfg.target_resolution * 111320

    if cfg.crs == "EPSG:6933":
        if trait_range <= cfg.target_resolution:
            log.warning(
                "Trait range of %.2f m is less than or equal to the existing map"
                "resolution of %.2f m. "
                "Using the map resolution for hexagon assignment...",
                trait_range,
                cfg.target_resolution,
            )
            trait_range = cfg.target_resolution

        log.info("Reprojecting coordinates to WGS84 for hexagon assignment...")
        # Convert coordinates to EPSG:4326 to get hexagon assignments
        meta = {
            trait_col: "float64",
            "x": "float64",
            "y": "float64",
            "lon": "float64",
            "lat": "float64",
        }
        trait_df = trait_df.map_partitions(
            reproject_xy_to_geo, from_crs=cfg.crs, meta=meta
        )
        # trait_df = reproject_xy_to_geo(trait_df, from_crs=cfg.crs)
        trait_df = trait_df.rename(
            columns={"x": "x_old", "y": "y_old", "lat": "y", "lon": "x"}
        )

    h3_res = acr_to_h3_res(trait_range)

    trait_df = assign_hexagons(trait_df, h3_res, dask=True).reset_index(drop=True)

    if cfg.crs == "EPSG:6933":
        # Revert back to the original coordinates
        trait_df = trait_df.drop(columns=["x", "y"]).rename(
            columns={"x_old": "x", "y_old": "y"}
        )

    if isinstance(trait_df, dd.DataFrame):
        log.info("Computing trait dask DataFrame...")
        trait_df = trait_df.compute()  # pyright: ignore[reportCallIssue]

    log.info("Assigning the best folds...")
    trait_df = assign_folds(
        trait_df,
        cfg.train.cv_splits.n_splits,
        cfg.train.cv_splits.n_sims,
        trait_col,
    )

    trait_df[["x", "y", "fold"]].drop_duplicates(
        subset=["x", "y"]
    ).reset_index(drop=True).to_parquet(splits_fn, compression="zstd")
    return None


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate spatial k-fold cross-validation splits."
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing splits"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Main function to generate spatial k-fold cross-validation splits."""
    syscfg = cfg[detect_system()][cfg.model_res]

    # Ignore warnings
    warnings.simplefilter(action="ignore", category=UserWarning)

    if args.debug:
        log.setLevel(logging.DEBUG)
        syscfg.skcv_splits.n_workers = 40

    log.info("Initializing Dask...")
    client, cluster = init_dask(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.skcv_splits.n_workers,
        # threads_per_worker=syscfg.skcv_splits.threads_per_worker,
    )
    ranges = pd.read_parquet(
        get_autocorr_ranges_fn(),
        columns=["trait", cfg.train.cv_splits.range_stat],
    )

    target_cols: pd.Index = dd.read_parquet(get_y_fn()).columns.difference(["source"])
    trait_cols: pd.Index = target_cols.difference(["x", "y"])

    traits = dd.read_parquet(get_y_fn(), columns=target_cols).repartition(
        npartitions=100
    )

    log.info("Assigning splits for traits...")
    # compute(
    #     *[
    #         _assign_trait_splits(traits, trait_col, ranges, args.overwrite, cfg)
    #         for trait_col in trait_cols
    #     ]
    # )
    for trait_col in trait_cols:
        _assign_trait_splits(traits, trait_col, ranges, args.overwrite, cfg)

    close_dask(client, cluster)
    log.info("Done!")


if __name__ == "__main__":
    main()

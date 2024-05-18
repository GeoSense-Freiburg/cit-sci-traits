"""Calculates spatial autocorrelation for each trait in a feature set."""

from pathlib import Path

import numpy as np
import pandas as pd
import utm
from box import ConfigBox
from dask import compute, delayed
from dask.distributed import Client
from pykrige.ok import OrdinaryKriging

from src.conf.conf import get_config
from src.conf.environment import log


@delayed
def get_utm_zones(x: np.ndarray, y: np.ndarray) -> tuple[list, list, list]:
    """
    Converts latitude and longitude coordinates to UTM zones.

    Args:
        x (np.ndarray): Array of longitude coordinates.
        y (np.ndarray): Array of latitude coordinates.

    Returns:
        tuple[list, list, list]: A tuple containing three lists - eastings, northings,
            and zones.
    """
    eastings, northings, zones = [], [], []

    for x_, y_ in zip(x, y):
        easting, northing, zone, letter = utm.from_latlon(y_, x_)
        eastings.append(easting)
        northings.append(northing)
        zones.append(f"{zone}{letter}")

    return eastings, northings, zones


def add_utm(df: pd.DataFrame, chunksize: int = 10000) -> pd.DataFrame:
    """
    Adds UTM coordinates to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to which UTM coordinates will be added.
        chunksize (int, optional): The size of each chunk for parallel processing.
            Defaults to 10000.

    Returns:
        pd.DataFrame: The DataFrame with UTM coordinates added.
    """
    x = df.x.to_numpy()
    y = df.y.to_numpy()

    # Split x and y into chunks
    x_chunks = [x[i : i + chunksize] for i in range(0, len(x), chunksize)]
    y_chunks = [y[i : i + chunksize] for i in range(0, len(y), chunksize)]

    # Compute the UTM zones for each chunk in parallel
    results = [
        get_utm_zones(x_chunk, y_chunk) for x_chunk, y_chunk in zip(x_chunks, y_chunks)
    ]

    results = compute(*results)

    # Assign the results to new columns in df
    df["easting"] = [e for result in results for e in result[0]]
    df["northing"] = [n for result in results for n in result[1]]
    df["zone"] = [z for result in results for z in result[2]]

    return df


@delayed
def calculate_variogram(group: pd.DataFrame, data_col: str, **kwargs) -> float | None:
    """
    Calculate the variogram for a given group of data points.

    Parameters:
        group (pd.DataFrame): The group of data points.
        data_col (str): The column name of the data points.
        **kwargs: Additional keyword arguments.

    Returns:
        float | None: The variogram value, or None if the group is not a DataFrame or
            has less than 200 rows.
    """
    if not isinstance(group, pd.DataFrame) or len(group) < 200:
        return 0

    n_max = 20_000

    if "n_max" in kwargs:
        n_max = kwargs.pop("n_max")

    if len(group) > n_max:
        group = group.sample(n_max)

    ok_vgram = OrdinaryKriging(
        group["easting"], group["northing"], group[data_col], **kwargs
    )

    return ok_vgram.variogram_model_parameters[1]


def main(cfg: ConfigBox = get_config()) -> None:
    """Main function for calculating spatial autocorrelation."""
    feats_fn = (
        Path(cfg.train.dir)
        / cfg.PFT
        / cfg.model_res
        / cfg.datasets.Y.use
        / cfg.train.features
    )

    log.info("Reading features from %s...", feats_fn)
    feats_cols = pd.read_parquet(feats_fn).columns
    trait_cols = [c for c in feats_cols if c.startswith("X")]

    for trait_col in trait_cols:
        log.info("Calculating spatial autocorrelation for %s...", trait_col)
        trait_df = (
            pd.read_parquet(feats_fn, columns=["x", "y", trait_col])
            .astype(np.float32)
            .dropna()
        )

        log.info("Adding UTM coordinates...")
        with Client(dashboard_address=cfg.dask_dashboard, n_workers=60):
            trait_df = add_utm(trait_df).drop(columns=["x", "y"])

        log.info("Calculating variogram ranges...")
        with Client(dashboard_address=cfg.dask_dashboard, n_workers=5):
            kwargs = {
                "n_max": 18000,
                "variogram_model": "spherical",
                "nlags": 15,
                "anisotropy_scaling": 1,
                "anisotropy_angle": 0,
            }

            results = [
                calculate_variogram(group, trait_col, **kwargs)
                for _, group in trait_df.groupby("zone")
            ]

            autocorr_ranges = list(compute(*results))

        filt_ranges = [r for r in autocorr_ranges if r != 0]

        log.info("Saving range statistics to DataFrame...")
        # Define the file path
        ranges_df_fn = (
            Path(cfg.train.dir)
            / cfg.PFT
            / cfg.model_res
            / cfg.datasets.Y.use
            / cfg.train.spatial_autocorr
        )

        # Try to read the existing DataFrame, or create a new one if it doesn't exist
        ranges_df = (
            pd.read_parquet(ranges_df_fn)
            if ranges_df_fn.exists()
            else pd.DataFrame(columns=["trait", "mean", "std", "median", "q05", "q95"])
        )

        # Create a new row and append it to the DataFrame
        ranges_df = pd.concat(
            [
                ranges_df,
                pd.DataFrame(
                    [
                        {
                            "trait": trait_col,
                            "mean": np.mean(filt_ranges),
                            "std": np.std(filt_ranges),
                            "median": np.median(filt_ranges),
                            "q05": np.quantile(filt_ranges, 0.05),
                            "q95": np.quantile(filt_ranges, 0.95),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        # Write the DataFrame back to disk
        ranges_df.to_parquet(ranges_df_fn)


if __name__ == "__main__":
    main()

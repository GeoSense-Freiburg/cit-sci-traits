"""Calculates spatial autocorrelation for each trait in a feature set."""

import dask.dataframe as dd
import numpy as np
import pandas as pd
import utm
from box import ConfigBox
from dask import compute, delayed
from pykrige.ok import OrdinaryKriging

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import get_autocorr_ranges_fn, get_y_fn


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
    # TODO: handle low res cases where many UTM zones have 1 or only a few points.
    # E.g. consider using web mercator in these instances...
    # TODO: Question: should we even be calculating spatial autorcorrelation separately
    # for each resolution? Or should we just calculate it once for the highest resolution?
    syscfg = cfg[detect_system()][cfg.model_res]

    y_fn = get_y_fn(cfg)

    log.info("Reading sPlot features from %s...", y_fn)
    y_ddf = (
        dd.read_parquet(y_fn)
        .pipe(lambda _ddf: _ddf[_ddf["source"] == "s"])
        .drop(columns=["source"])
    )
    y_cols = y_ddf.columns.difference(["x", "y"]).to_list()

    for i, trait_col in enumerate(y_cols):
        log.info("Calculating spatial autocorrelation for %s...", trait_col)
        trait_df = (
            y_ddf[["x", "y", trait_col]]
            .astype(np.float32)
            .compute()
            .reset_index(drop=True)
        )

        log.info("Adding UTM coordinates...")
        client, cluster = init_dask(
            dashboard_address=cfg.dask_dashboard,
            n_workers=syscfg.calc_spatial_autocorr.n_workers,
        )

        trait_df = add_utm(trait_df).drop(columns=["x", "y"])

        close_dask(client, cluster)

        log.info("Calculating variogram ranges...")
        client, cluster = init_dask(
            dashboard_address=cfg.dask_dashboard,
            n_workers=syscfg.calc_spatial_autocorr.n_workers_variogram,
        )

        kwargs = {
            # "n_max": 18000,
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

        close_dask(client, cluster)

        filt_ranges = [r for r in autocorr_ranges if r != 0]

        log.info("Saving range statistics to DataFrame...")
        # Define the file path
        ranges_fn = get_autocorr_ranges_fn(cfg)

        # Try to read the existing DataFrame, or create a new one if this is the first trait
        ranges_df = (
            pd.read_parquet(ranges_fn)
            if i > 0
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
        ranges_df.to_parquet(ranges_fn)


if __name__ == "__main__":
    main()

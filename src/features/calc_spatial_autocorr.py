"""Calculates spatial autocorrelation for each trait in a feature set."""

import shutil
from pathlib import Path

import dask.dataframe as dd
import geopandas as gpd
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
def calculate_variogram_pykrige(
    group: pd.DataFrame, data_col: str, **kwargs
) -> tuple[float | None, int]:
    """
    Calculate the variogram for a given group of data points.

    Parameters:
        group (pd.DataFrame): The group of data points.
        data_col (str): The column name of the data points.
        **kwargs: Additional keyword arguments.

    Returns:
        float | None: The variogram value, or None if the group is not a DataFrame or
            has less than 200 rows.
        int: The number of samples used to calculate the variogram.
    """
    if not isinstance(group, pd.DataFrame) or len(group) < 200:
        return 0, 0

    n_max = 20_000

    if "n_max" in kwargs:
        n_max = kwargs.pop("n_max")

    group = group.copy()

    if len(group) > n_max:
        group = group.sample(n_max)

    n_samples = len(group)

    ok_vgram = OrdinaryKriging(
        group["easting"], group["northing"], group[data_col], **kwargs
    )

    return ok_vgram.variogram_model_parameters[1], n_samples


def copy_ref_to_dvc(cfg: ConfigBox) -> None:
    """Copy the reference ranges file to the DVC-tracked location."""
    fn = (
        f"{Path(cfg.train.spatial_autocorr).stem }_"
        f"{cfg.calc_spatial_autocorr.use_existing}"
        f"{Path(cfg.train.spatial_autocorr).suffix}"
    )
    ranges_fn_ref = Path("reference", fn)
    log.info("Using existing spatial autocorrelation ranges from %s...", ranges_fn_ref)
    ranges_fn_dvc = get_autocorr_ranges_fn(cfg)

    if ranges_fn_dvc.exists():
        log.info("Overwriting existing spatial autocorrelation ranges...")
        ranges_fn_dvc.unlink()
    shutil.copy(ranges_fn_ref, ranges_fn_dvc)


def main(cfg: ConfigBox = get_config()) -> None:
    """Main function for calculating spatial autocorrelation."""
    syscfg = cfg[detect_system()][cfg.model_res]["calc_spatial_autocorr"]

    if cfg.calc_spatial_autocorr.use_existing:
        copy_ref_to_dvc(cfg)
        return

    y_fn = get_y_fn(cfg)

    # Use only sPlot data to calculate spatial autocorrelation
    log.info("Reading sPlot features from %s...", y_fn)
    y_ddf = dd.read_parquet(y_fn).query("source == 's'").drop(columns=["source"])
    y_cols = y_ddf.columns.difference(["x", "y"]).to_list()

    vgram_kwargs = {
        "n_max": 18000,
        "variogram_model": "spherical",
        "nlags": 15,
        "anisotropy_scaling": 1,
        "anisotropy_angle": 0,
    }

    for i, trait_col in enumerate(y_cols):
        log.info("Calculating spatial autocorrelation for %s...", trait_col)
        trait_df = (
            y_ddf[["x", "y", trait_col]]
            .astype(np.float32)
            .compute()
            .reset_index(drop=True)
        )

        if cfg.target_resolution > 0.2 and cfg.crs == "EPSG:4326":
            log.info(
                "Target resolution of > 0.2 deg detected. Using web mercator "
                "coordinates instead of UTM zones..."
            )
            trait_df_wmerc = (
                gpd.GeoDataFrame(  # pyright: ignore[reportCallIssue]
                    trait_df.drop(columns=["x", "y"]),
                    geometry=gpd.points_from_xy(trait_df.x, trait_df.y),
                    crs="EPSG:4326",
                )
                .to_crs("EPSG:3857")
                .pipe(  # pyright: ignore[reportOptionalMemberAccess]
                    # Technically not easting/northing but it's needed to work w/ vgram fn
                    lambda _df: _df.assign(
                        easting=_df.geometry.x, northing=_df.geometry.y
                    ).drop(columns=["geometry"])
                )
            )

            log.info("Calculating variogram ranges...")
            client, cluster = init_dask(
                dashboard_address=cfg.dask_dashboard,
                n_workers=syscfg.n_workers_variogram,
            )
            results = [
                calculate_variogram_pykrige(trait_df_wmerc, trait_col, **vgram_kwargs)
            ]

        elif cfg.crs == "EPSG:4326":
            log.info("Adding UTM coordinates...")
            client, cluster = init_dask(
                dashboard_address=cfg.dask_dashboard,
                n_workers=syscfg.n_workers,
            )

            trait_df = add_utm(trait_df).drop(columns=["x", "y"])

            close_dask(client, cluster)

            log.info("Calculating variogram ranges...")
            client, cluster = init_dask(
                dashboard_address=cfg.dask_dashboard,
                n_workers=syscfg.n_workers_variogram,
            )

            results = [
                calculate_variogram_pykrige(group, trait_col, **vgram_kwargs)
                for _, group in trait_df.groupby("zone")
            ]

        elif cfg.crs == "EPSG:6933":

            def _assign_zones(df: pd.DataFrame, n_zones: int) -> pd.DataFrame:
                """
                Assigns zones to the DataFrame based on x and y coordinates.

                Args:
                    df (pd.DataFrame): The DataFrame with x and y coordinates.
                    n_sectors (int): The number of sectors to divide the data into.

                Returns:
                    pd.DataFrame: The DataFrame with an additional 'zone' column.
                """

                x_bins = np.linspace(df.easting.min(), df.easting.max(), n_zones + 1)
                y_bins = np.linspace(
                    df.northing.min(), df.northing.max(), n_zones // 2 + 1
                )

                x_zones = np.digitize(df.easting, x_bins) - 1
                y_zones = np.digitize(df.northing, y_bins) - 1

                df["zone"] = [f"{x}_{y}" for x, y in zip(x_zones, y_zones)]
                return df

            # Convert x and y to easting and northing (simple shift) as EPSG:6933 contains
            # negative x and y coordinates
            trait_df = trait_df.assign(
                easting=trait_df.x + abs(trait_df.x.min()),
                northing=trait_df.y + abs(trait_df.y.min()),
            ).drop(columns=["x", "y"])

            if syscfg.n_chunks > 1:
                trait_df_grouped = trait_df.pipe(
                    _assign_zones, n_zones=syscfg.n_chunks
                ).groupby("zone")

                log.info("Calculating variogram ranges...")
                client, cluster = init_dask(
                    dashboard_address=cfg.dask_dashboard,
                    # n_workers=syscfg.n_chunks // 2,
                    n_workers=1,
                    # threads_per_worker=1,
                )

                vgram_kwargs["n_max"] = len(trait_df)
                results = [
                    calculate_variogram_pykrige(group, trait_col)
                    for _, group in trait_df_grouped
                ]
            else:
                log.info("Calculating variogram ranges...")
                client, cluster = init_dask(
                    dashboard_address=cfg.dask_dashboard,
                    n_workers=1,
                )

                vgram_kwargs["n_max"] = len(trait_df)

                results = [
                    calculate_variogram_pykrige(trait_df, trait_col, **vgram_kwargs)
                ]

        else:
            raise ValueError(f"Unknown CRS: {cfg.crs}")

        autocorr_ranges = list(compute(*results))

        close_dask(client, cluster)

        filt_ranges = [(n, r) for n, r in autocorr_ranges if n > 0]
        log.info("Saving range statistics to DataFrame...")

        # Path to be checked into DVC
        ranges_fn_dvc = get_autocorr_ranges_fn(cfg)

        # Path to be used as reference when computing ranges for other resolutions.
        # Tracked with git.
        ranges_fn_ref = Path(
            "reference", f"{ranges_fn_dvc.stem}_{cfg.model_res}{ranges_fn_dvc.suffix}"
        )

        # Try to read the existing DataFrame, or create a new one if this is the first trait
        ranges_df = (
            pd.read_parquet(ranges_fn_dvc)
            if i > 0
            else pd.DataFrame(columns=["trait", "mean", "std", "median", "q05", "q95"])
        )

        # Weight the ranges by the number of samples used to calculate them
        # The weights and ranges are in a list of tuples, where the first element is the
        # number of samples and the second element is the range. We want to normalize the
        # sample_sizes so that they sum to 1, effectively converting them to weights
        sample_sizes = np.array([n for n, _ in filt_ranges])
        weights = sample_sizes / sample_sizes.sum()
        ranges = np.array([r for _, r in filt_ranges])

        # Create a new row and append it to the DataFrame
        ranges_df = pd.concat(
            [
                ranges_df,
                pd.DataFrame(
                    [
                        {
                            "trait": trait_col,
                            "mean": np.average(ranges, weights=weights),
                            "std": np.sqrt(
                                np.average(
                                    (ranges - ranges.mean()) ** 2, weights=weights
                                )
                            ),
                            "median": np.median(ranges),
                            "q05": np.quantile(ranges, 0.05),
                            "q95": np.quantile(ranges, 0.95),
                            "n": sample_sizes.sum(),
                            "n_chunks": len(ranges),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        # Write the DataFrame back to disk
        ranges_df.to_parquet(ranges_fn_dvc)
        ranges_df.to_parquet(ranges_fn_ref)


if __name__ == "__main__":
    main()

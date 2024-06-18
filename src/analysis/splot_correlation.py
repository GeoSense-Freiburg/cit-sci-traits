"""Calculate Pearson correlation coefficient between extrapolated trait maps and sparse
sPlot grids."""

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import statsmodels.api as sm
from box import ConfigBox
from dask import compute, delayed
from dask.distributed import Client
from pyproj import Proj
from shapely.geometry import shape

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.raster_utils import open_raster


def cli() -> argparse.Namespace:
    """Parse command-line interface arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--n_procs",
        type=int,
        default=-1,
        help="Number of processors to use for parallel processing",
    )

    args = parser.parse_args()

    if args.n_procs == -1:
        args.n_procs = None

    return args


def get_lat_area(lat: int | float, resolution: int | float) -> float:
    """Calculate the area of a grid cell at a given latitude."""
    # Define the grid cell coordinates
    coordinates = [
        (0, lat + (resolution / 2)),
        (resolution, lat + (resolution / 2)),
        (resolution, lat - (resolution / 2)),
        (0, lat - (resolution / 2)),
        (0, lat + (resolution / 2)),  # Close the polygon by repeating the first point
    ]

    # Define the projection string directly using the coordinates
    projection_string = (
        f"+proj=aea +lat_1={coordinates[0][1]} +lat_2={coordinates[2][1]} "
        f"+lat_0={lat} +lon_0={resolution / 2}"
    )
    pa = Proj(projection_string)

    # Project the coordinates and create the polygon
    x, y = pa(*zip(*coordinates))  # pylint: disable=unpacking-non-sequence
    area = shape({"type": "Polygon", "coordinates": [list(zip(x, y))]}).area / 1000000

    return area


def lat_weights(lat_unique: Iterable[int | float], resolution: int | float) -> dict:
    """Calculate weights for each latitude band based on area of grid cells."""
    weights = {}

    for j in lat_unique:
        weights[j] = get_lat_area(j, resolution)

    # Normalize the weights by the maximum area
    max_area = max(weights.values())
    weights = {k: v / max_area for k, v in weights.items()}

    return weights


def weighted_pearson_r(df: pd.DataFrame, weights: dict) -> float:
    """Calculate the weighted Pearson correlation coefficient between two DataFrames."""
    df["weights"] = df.index.get_level_values("y").map(weights)
    model = sm.stats.DescrStatsW(df.iloc[:, :2], df["weights"])
    return model.corrcoef[0, 1]


def get_fns(cfg: ConfigBox) -> tuple[list[Path], list[Path]]:
    """Get the filenames of the sPlot and extrapolated trait maps."""
    splot_fns = sorted(
        list(
            Path(
                cfg.interim_dir,
                cfg.splot.interim.dir,
                cfg.splot.interim.traits,
                cfg.PFT,
                cfg.model_res,
            ).glob("*.tif")
        ),
        key=lambda x: int(x.stem.split("X")[-1]),
    )
    extrap_fns = sorted(
        list(
            Path(
                cfg.processed.dir,
                cfg.PFT,
                cfg.model_res,
                cfg.datasets.Y.use,
                cfg.processed.predict_dir,
            ).glob("*.tif")
        ),
        key=lambda x: int(x.stem.split("_")[0].split("X")[-1]),
    )

    return splot_fns, extrap_fns


@delayed
def process_pair(fn1: Path, fn2: Path, resolution: int | float) -> tuple[str, float]:
    """Calculate the weighted Pearson correlation coefficient between a pair of trait maps."""
    log.info("Loading and filtering data for %s...", fn2.stem)
    df1 = (
        open_raster(fn1)
        .sel(band=1)
        .to_dataframe(name=f"left_{fn1.stem}")
        .drop(columns=["band", "spatial_ref"])
        .dropna()
    )
    df2 = (
        open_raster(fn2)
        .sel(band=1)
        .to_dataframe(name=f"right_{fn2.stem}")
        .drop(columns=["band", "spatial_ref"])
        .dropna()
    )
    log.info("Joining dataframes (%s)...", fn2.stem)
    df = df1.join(df2, how="inner")

    lat_unique = df.index.get_level_values("y").unique()

    log.info("Calculating weights (%s)...", fn2.stem)
    weights = lat_weights(lat_unique, resolution)

    log.info("Calculating weighted Pearson correlation coefficient (%s)...", fn2.stem)
    r = weighted_pearson_r(df, weights)

    log.info("Weighted Pearson correlation coefficient: %s", r)

    return fn2.stem, r


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Main function"""
    splot_fns, extrap_fns = get_fns(cfg)

    log.info("Processing %s pairs of trait maps...", len(splot_fns))
    with Client(dashboard_address=cfg.dask_dashboard, n_workers=args.n_procs):
        delayed_results = [
            process_pair(splot_fn, extrap_fn, cfg.target_resolution)
            for splot_fn, extrap_fn in zip(splot_fns, extrap_fns)
        ]

        log.info("Computing results...")
        results = compute(*delayed_results)

    log.info("Writing results to disk...")

    pd.DataFrame(results, columns=["trait", "r"]).sort_values("trait").to_csv(
        Path(
            cfg.processed.dir,
            cfg.PFT,
            cfg.model_res,
            cfg.datasets.Y.use,
            cfg.processed.splot_corr,
        ),
        index=False,
    )


if __name__ == "__main__":
    main()

"""Calculate Pearson correlation coefficient between extrapolated trait maps and sparse
sPlot grids."""

import argparse
from pathlib import Path

import pandas as pd
from box import ConfigBox
from dask import compute, delayed
from dask.distributed import Client

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.raster_utils import open_raster
from src.utils.spatial_utils import lat_weights
from src.utils.spatial_utils import weighted_pearson_r


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

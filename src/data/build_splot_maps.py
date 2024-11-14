""""Match sPlot data with filtered trait data, calculate CWMs, and grid it."""

import argparse
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.df_utils import rasterize_points, reproject_geo_to_xy
from src.utils.raster_utils import xr_to_raster
from src.utils.trait_utils import (
    clean_species_name,
    filter_pft,
    get_trait_number_from_id,
)


def _cw_stats(g: pd.DataFrame, col: str) -> pd.Series:
    """Calculate all community-weighted stats per plot."""
    # Normalize the abundances to sum to 1. Important when not all species in a plot are
    # present in the trait data.
    normalized_abund = g["Rel_Abund_Plot"] / g["Rel_Abund_Plot"].sum()
    if g.empty:
        log.warning("Empty group detected, returning NaNs...")
        return pd.Series(
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            index=["cwm", "cw_std", "cw_med", "cw_q05", "cw_q95"],
        )
    return pd.Series(
        [
            _cwm(g[col], normalized_abund),
            _cw_std(g[col], normalized_abund),
            _cw_quantile(g[col].to_numpy(), normalized_abund.to_numpy(), 0.5),
            _cw_quantile(g[col].to_numpy(), normalized_abund.to_numpy(), 0.05),
            _cw_quantile(g[col].to_numpy(), normalized_abund.to_numpy(), 0.95),
        ],
        index=["cwm", "cw_std", "cw_med", "cw_q05", "cw_q95"],
    )


def _cwm(data: pd.Series, weights: pd.Series) -> float | Any:
    """Calculate the community-weighted mean."""
    return np.average(data, weights=weights)


def _cw_std(data: pd.Series, weights: pd.Series) -> float:
    """Calculate the community-weighted standard deviation."""
    return np.sqrt(np.average((data - data.mean()) ** 2, weights=weights))


def _cw_quantile(data: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Calculate the community-weighted quantile."""
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cumsum = np.cumsum(sorted_weights)
    quantile_value = sorted_data[cumsum >= quantile][0]
    return quantile_value


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Match sPlot data with filtered trait data, calculate CWMs, and grid it."
    )
    parser.add_argument(
        "-r", "--resume", action="store_true", help="Resume from last run."
    )
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Match sPlot data with filtered trait data, calculate CWMs, and grid it."""
    sys_cfg = cfg[detect_system()][cfg.model_res]["build_splot_maps"]
    # Setup ################
    splot_dir = (
        Path(cfg.interim_dir, cfg.splot.interim.dir) / cfg.splot.interim.extracted
    )

    def _repartition_if_set(df: dd.DataFrame, npartitions: int | None) -> dd.DataFrame:
        return (
            df.repartition(npartitions=npartitions) if npartitions is not None else df
        )

    # create dict of dask kws, but only if they are not None
    dask_kws = {k: v for k, v in sys_cfg.dask.items() if v is not None}
    client, cluster = init_dask(dashboard_address=cfg.dask_dashboard, **dask_kws)
    # /Setup ################

    # Load header and set plot IDs as index for later joining with vegetation data
    header = (
        dd.read_parquet(
            splot_dir / "header.parquet",
            columns=["PlotObservationID", "Longitude", "Latitude"],
        )
        .pipe(_repartition_if_set, sys_cfg.npartitions)
        .astype({"Longitude": np.float64, "Latitude": np.float64})
        .set_index("PlotObservationID")
        .map_partitions(reproject_geo_to_xy, crs=cfg.crs, x="Longitude", y="Latitude")
        .drop(columns=["Longitude", "Latitude"])
    )

    # Load pre-cleaned and filtered TRY traits and set species as index
    traits = (
        dd.read_parquet(
            Path(cfg.interim_dir, cfg.trydb.interim.dir) / cfg.trydb.interim.filtered
        )
        .pipe(_repartition_if_set, sys_cfg.npartitions)
        .set_index("speciesname")
    )

    # Load PFT data, filter by desired PFT, clean species names, and set them as index
    # for joining
    pfts = (
        dd.read_csv(Path(cfg.raw_dir) / cfg.trydb.raw.pfts, encoding="latin-1")
        .pipe(_repartition_if_set, sys_cfg.npartitions)
        .pipe(filter_pft, cfg.PFT)
        .drop(columns=["AccSpeciesID"])
        .dropna(subset=["AccSpeciesName"])
        .pipe(clean_species_name, "AccSpeciesName", "speciesname")
        .drop(columns=["AccSpeciesName"])
        .set_index("speciesname")
    )

    # Load sPlot vegetation records, clean species names, match with desired PFT, and
    # merge with trait data
    merged = (
        dd.read_parquet(
            splot_dir / "vegetation.parquet",
            columns=[
                "PlotObservationID",
                "Species",
                "Rel_Abund_Plot",
            ],
        )
        .pipe(_repartition_if_set, sys_cfg.npartitions)
        .dropna(subset=["Species"])
        .pipe(clean_species_name, "Species", "speciesname")
        .drop(columns=["Species"])
        .set_index("speciesname")
        .join(pfts, how="inner")
        .join(traits, how="inner")
        .reset_index()
        .drop(columns=["pft", "speciesname"])
        .persist()
    )

    out_dir = (
        Path(cfg.interim_dir, cfg.splot.interim.dir)
        / cfg.splot.interim.traits
        / cfg.PFT
        / cfg.model_res
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [col for col in merged.columns if col.startswith("X")]
    valid_traits = [str(trait_num) for trait_num in cfg.datasets.Y.traits]
    cols = [col for col in cols if get_trait_number_from_id(col) in valid_traits]

    try:
        for col in cols:
            out_path = out_dir / f"{col}.tif"
            if args.resume and out_path.exists():
                log.info("%s.tif already exists, skipping...", col)
                continue

            log.info("Processing trait %s...", col)
            # Calculate community-weighted means per plot, join with `header` to get
            # plot lat/lons, and grid at the configured resolution.
            df = (
                merged[["PlotObservationID", "Rel_Abund_Plot", col]]
                .set_index("PlotObservationID")
                .groupby("PlotObservationID")
                .apply(
                    _cw_stats,
                    col,
                    meta={
                        "cwm": "f8",
                        "cw_std": "f8",
                        "cw_med": "f8",
                        "cw_q05": "f8",
                        "cw_q95": "f8",
                    },
                )
                .join(header, how="inner")
                .reset_index(drop=True)
                .compute()
            )

            mean = {"mean": "mean"}
            mean_and_count = {"mean": "mean", "count": "count"}
            stat_cols = ["cwm", "cw_std", "cw_med", "cw_q05", "cw_q95"]
            stat_names = ["mean", "std", "median", "q05", "q95"]

            grids = []
            for stat_col, stat_name in zip(stat_cols, stat_names):
                log.info("Rasterizing %s...", stat_col)
                funcs = mean
                if stat_col == "cwm":
                    funcs = mean_and_count

                ds = rasterize_points(
                    df,
                    data=stat_col,
                    res=cfg.target_resolution,
                    crs=cfg.crs,
                    agg=True,
                    funcs=funcs,
                )

                if stat_col != "cwm":
                    ds = ds.rename({"mean": stat_name})

                grids.append(ds)

            # Merge all of the gridded data into a single dataset
            log.info("Merging gridded data...")
            gridded_trait = xr.merge(grids)

            out_fn = out_dir / f"{col}.tif"
            log.info("Writing %s to disk...", col)
            xr_to_raster(gridded_trait, out_fn)
            log.info("Wrote %s.", out_fn)
    finally:
        close_dask(client, cluster)
        log.info("Done!")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

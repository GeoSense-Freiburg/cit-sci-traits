""""Match sPlot data with filtered trait data, calculate CWMs, and grid it."""

import argparse
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.df_utils import rasterize_points, reproject_geo_to_xy
from src.utils.raster_utils import xr_to_raster
from src.utils.trait_utils import clean_species_name, filter_pft


def _cwm(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Calculate community-weighted means per plot."""
    grouped = df.groupby("PlotObservationID")

    result = grouped.apply(
        lambda g: pd.Series(
            [
                np.average(g[col], weights=g["Rel_Abund_Plot"]),
            ],
            index=["cwm"],
        ),
        # meta={"cwm": "f8"},
    )

    return result


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
        .map_partitions(reproject_geo_to_xy, crs=cfg.crs, x="Longitude", y="Latitude")
        .drop(columns=["Longitude", "Latitude"])
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

    try:
        for col in cols:
            out_path = out_dir / f"{col}.tif"
            if args.resume and out_path.exists():
                log.info("%s.tif already exists, skipping...", col)
                continue

            log.info("Processing trait %s...", col)
            # Calculate community-weighted means per plot, join with `header` to get
            # plot lat/lons, and grid at the configured resolution.
            gridded_cwms = (
                merged[["PlotObservationID", "Rel_Abund_Plot", col]]
                .set_index("PlotObservationID")
                .map_partitions(_cwm, col, meta={"cwm": "f8"})
                .join(header, how="inner")
                .reset_index()
                .pipe(
                    rasterize_points, data="cwm", res=cfg.target_resolution, crs=cfg.crs
                )
            )

            out_fn = out_dir / f"{col}.tif"
            log.info("Writing %s to disk...", col)
            xr_to_raster(gridded_cwms, out_fn)
            log.info("Wrote %s.", out_fn)
    finally:
        close_dask(client, cluster)
        log.info("Done!")


if __name__ == "__main__":
    main()

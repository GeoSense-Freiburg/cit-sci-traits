""""Match sPlot data with filtered trait data, calculate CWMs, and grid it."""

from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from box import ConfigBox
from dask.distributed import Client, LocalCluster

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.df_utils import global_grid_df, grid_df_to_raster
from src.utils.trait_utils import clean_species_name, filter_pft


def cwm(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Calculate community-weighted means per plot."""
    grouped = df.groupby("PlotObservationID")

    result = grouped.apply(
        lambda g: pd.Series(
            [
                np.average(g[col], weights=g["Rel_Abund_Plot"]),
            ],
            index=["cwm"],
        )
    )

    return result


def main(cfg: ConfigBox = get_config()) -> None:
    """Match sPlot data with filtered trait data, calculate CWMs, and grid it."""

    # Setup ################
    splot_dir = (
        Path(cfg.interim_dir, cfg.splot.interim.dir) / cfg.splot.interim.extracted
    )
    npartitions = 60
    cluster = LocalCluster(
        n_workers=40, memory_limit="40GB", dashboard_address=":39143"
    )
    client = Client(cluster)
    # /Setup ################

    # Load header and set plot IDs as index for later joining with vegetation data
    header = (
        dd.read_parquet(
            splot_dir / "header.parquet",
            columns=["PlotObservationID", "Longitude", "Latitude"],
        )
        .repartition(npartitions=npartitions)
        .astype({"Longitude": np.float64, "Latitude": np.float64})
        .set_index("PlotObservationID")
    )

    # Load pre-cleaned and filtered TRY traits and set species as index
    traits = (
        dd.read_parquet(
            Path(cfg.interim_dir, cfg.trydb.interim.dir) / cfg.trydb.interim.filtered
        )
        .repartition(npartitions=npartitions)
        .set_index("speciesname")
    )

    # Load PFT data, filter by desired PFT, clean species names, and set them as index
    # for joining
    pfts = (
        dd.read_csv(cfg.trydb.raw.pfts, encoding="latin-1")
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
        .repartition(npartitions=npartitions)
        .dropna(subset=["Species"])
        .pipe(clean_species_name, "Species", "speciesname")
        .drop(columns=["Species"])
        .set_index("speciesname")
        .join(pfts, how="inner")
        .join(traits, how="inner")
        .reset_index()
        .drop(columns=["pft", "speciesname"])
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
            log.info("Processing trait %s...", col)
            # Calculate community-weighted means per plot, join with `header` to get
            # plot lat/lons, and grid at the configured resolution.
            gridded_cwms = (
                merged[["PlotObservationID", "Rel_Abund_Plot", col]]
                .set_index("PlotObservationID")
                .persist()
                .map_partitions(cwm, col, meta={"cwm": "f8"})
                .join(header, how="inner")
                .reset_index()
                .pipe(
                    global_grid_df,
                    "cwm",
                    "Longitude",
                    "Latitude",
                    cfg.target_resolution,
                )
                .compute()
            )

            grid_df_to_raster(
                gridded_cwms, cfg.target_resolution, out_dir / f"{col}.tif"
            )
            log.info("Wrote %s.tif.", col)
    finally:
        client.close()
        cluster.close()
        log.info("Done!")


if __name__ == "__main__":
    main()

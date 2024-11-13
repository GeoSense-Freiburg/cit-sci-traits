"""
Match subsampled GBIF data with filtered trait data, grid it, generate grid cell
statistics, and write each trait's corresponding raster stack to GeoTIFF files.
"""

import argparse
from pathlib import Path

import dask.dataframe as dd
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.df_utils import rasterize_points, reproject_geo_to_xy
from src.utils.raster_utils import xr_to_raster
from src.utils.trait_utils import filter_pft, get_trait_number_from_id


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Match subsampled GBIF data with filtered trait data, grid it, generate grid cell statistics, and write each trait's corresponding raster stack to GeoTIFF files."
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing files."
    )
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Main function."""
    syscfg = cfg[detect_system()][cfg.model_res]["build_gbif_maps"]

    # Initialize Dask client
    log.info("Initializing Dask client...")
    client, cluster = init_dask(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.n_workers,
        memory_limit=syscfg.memory_limit,
    )

    out_dir = (
        Path(cfg.interim_dir)
        / cfg.gbif.interim.dir
        / cfg.gbif.interim.traits
        / cfg.PFT
        / cfg.model_res
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    gbif = (
        dd.read_parquet(
            Path(cfg.interim_dir, cfg.gbif.interim.dir, cfg.gbif.interim.matched)
        )
        .pipe(filter_pft, cfg.PFT)
        .set_index("speciesname")
    )

    mn_traits = dd.read_parquet(
        Path(cfg.interim_dir, cfg.trydb.interim.dir, cfg.trydb.interim.filtered)
    ).set_index("speciesname")

    # Merge GBIF and trait data
    merged = (
        gbif.join(mn_traits, how="inner").reset_index(drop=True).drop(columns=["pft"])
    )

    # Reproject coordinates to target CRS
    if cfg.crs == "EPSG:6933":
        merged = merged.map_partitions(
            reproject_geo_to_xy,
            to_crs=cfg.crs,
            x="decimallongitude",
            y="decimallatitude",
        ).drop(columns=["decimallatitude", "decimallongitude"])

    # Grid trait stats (mean, STD, median, 5th and 95th quantiles) for each grid cell
    cols = [col for col in merged.columns if col.startswith("X")]
    valid_traits = [str(trait_num) for trait_num in cfg.datasets.Y.traits]
    cols = [col for col in cols if get_trait_number_from_id(col) in valid_traits]

    try:
        for col in cols:
            out_fn = out_dir / f"{col}.tif"
            if out_fn.exists() and not args.overwrite:
                log.info("%s.tif already exists. Skipping...", col)
                continue

            log.info("Processing trait %s...", col)
            raster = rasterize_points(
                merged[["x", "y", col]],
                data=str(col),
                res=cfg.target_resolution,
                crs=cfg.crs,
                n_min=cfg.gbif.interim.min_count,
                n_max=cfg.gbif.interim.max_count,
            )

            log.info("Writing to disk...")
            xr_to_raster(raster, out_fn)
            log.info("Wrote %s.tif.", col)
    finally:
        close_dask(client, cluster)
        log.info("Done!")


if __name__ == "__main__":
    main()

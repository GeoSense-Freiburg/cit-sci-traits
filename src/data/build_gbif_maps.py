"""
Match subsampled GBIF data with filtered trait data, grid it, generate grid cell
statistics, and write each trait's corresponding raster stack to GeoTIFF files.
"""

from pathlib import Path

import dask.dataframe as dd
from box import ConfigBox
from dask.distributed import Client, LocalCluster

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.df_utils import global_grid_df, grid_df_to_raster
from src.utils.trait_utils import filter_pft


def main(cfg: ConfigBox = get_config()) -> None:
    """Main function."""
    # Initialize Dask client
    log.info("Initializing Dask client...")
    cluster = LocalCluster(
        n_workers=50, memory_limit="24GB", dashboard_address=":39143"
    )
    client = Client(cluster)
    npartitions = 90

    out_dir = (
        Path(cfg.train.dir)
        / cfg.train.traits.dir_name
        / cfg.PFT
        / cfg.model_res
        / cfg.train.gbif.dir_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    gbif = (
        dd.read_parquet(Path(cfg.gbif.interim.dir, cfg.gbif.interim.subsampled))
        .repartition(npartitions=npartitions)
        .pipe(filter_pft, cfg.PFT)
        .set_index("speciesname")
    )

    mn_traits = (
        dd.read_parquet(Path(cfg.trydb.interim.dir, cfg.trydb.interim.filtered))
        .repartition(npartitions=npartitions)
        .set_index("speciesname")
    )

    # Merge GBIF and trait data
    merged = gbif.join(mn_traits, how="inner").reset_index().drop(columns=["pft"])

    # Grid trait stats (mean, STD, median, 5th and 95th quantiles) for each grid cell
    cols = [col for col in merged.columns if col.startswith("X")]

    try:
        for col in cols:
            log.info("Processing trait %s...", col)
            grid_data = global_grid_df(merged, col, res=cfg.target_resolution).compute()
            grid_df_to_raster(grid_data, cfg.target_resolution, out_dir / f"{col}.tif")
            log.info("Wrote %s.tif.", col)
    finally:
        client.close()
        cluster.close()
        log.info("Done!")


if __name__ == "__main__":
    main()

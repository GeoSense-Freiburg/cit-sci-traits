"""Featurize training data."""

from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import xarray as xr
from box import ConfigBox
from dask import config
from dask.distributed import Client

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dataset_utils import (
    check_y_set,
    compute_partitions,
    get_trait_map_fns,
    get_y_fn,
    load_rasters_parallel,
)


def ds_to_ddf(ds: xr.Dataset) -> dd.DataFrame:
    """Convert an xarray Dataset to a dask DataFrame"""
    return (
        ds.to_dask_dataframe()
        .drop(columns=["band", "spatial_ref"])
        .pipe(
            lambda _ddf: _ddf.dropna(
                how="all", subset=_ddf.columns.difference(["x", "y"])
            )
        )
    )


def build_y_df(
    fns: list[Path], cfg: ConfigBox, syscfg: ConfigBox, trait_set: str
) -> pd.DataFrame:
    """Build dataframe of Y data for a given trait set."""
    check_y_set(trait_set)
    log.info("Loading Y data (%s)...", trait_set)
    y_ds = load_rasters_parallel(
        fns, cfg.datasets.Y.trait_stat, syscfg.featurize_train.n_chunks
    )

    log.info("Computing Y data (%s)...", trait_set)
    y_ddf = ds_to_ddf(y_ds)
    y_df = compute_partitions(y_ddf).reset_index(drop=True).assign(source=trait_set[0])

    y_ds.close()
    return y_df


def main(cfg: ConfigBox = get_config()) -> None:
    """
    Main function for featurizing training data.

    Args:
        args (argparse.Namespace): Command-line arguments.
        cfg (ConfigBox): Configuration settings.

    Returns:
        None
    """
    syscfg = cfg[detect_system()][cfg.model_res]

    with Client(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.featurize_train.n_workers,
        memory_limit=syscfg.featurize_train.memory_limit,
    ), config.set({"array.slicing.split_large_chunks": False}):

        gbif_trait_map_fns = get_trait_map_fns("gbif", cfg)
        splot_trait_map_fns = get_trait_map_fns("splot", cfg)

        log.info("Combining sPlot and GBIF...")
        y_df = pd.concat(
            [
                build_y_df(gbif_trait_map_fns, cfg, syscfg, "gbif"),
                build_y_df(splot_trait_map_fns, cfg, syscfg, "splot"),
            ],
            axis=0,
            ignore_index=True,
        )

        out_path = get_y_fn(cfg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Writing Y data to %s...", str(out_path))
        y_df.to_parquet(out_path, compression="zstd")


if __name__ == "__main__":
    main()

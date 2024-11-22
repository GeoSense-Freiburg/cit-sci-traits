"""Featurize training data."""

from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import xarray as xr
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import (
    check_y_set,
    get_trait_map_fns,
    get_y_fn,
    load_rasters_parallel,
)
from src.utils.trait_utils import get_trait_number_from_id


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
    y_ds = load_rasters_parallel(fns, cfg.datasets.Y.trait_stat, syscfg.n_chunks)

    log.info("Computing Y data (%s)...", trait_set)
    y_ddf = ds_to_ddf(y_ds).assign(source=trait_set[0]).astype({"source": "category"})

    y_ds.close()
    return y_ddf


def main(cfg: ConfigBox = get_config()) -> None:
    """
    Main function for featurizing training data.

    Args:
        args (argparse.Namespace): Command-line arguments.
        cfg (ConfigBox): Configuration settings.

    Returns:
        None
    """
    syscfg = cfg[detect_system()][cfg.model_res]["featurize_train"]

    log.info("Initializing Dask client...")
    client, cluster = init_dask(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.n_workers,
        threads_per_worker=syscfg.threads_per_worker,
    )

    log.info("Gathering trait map filenames...")
    valid_traits = [str(trait_num) for trait_num in cfg.datasets.Y.traits]
    gbif_trait_map_fns = [
        fn
        for fn in get_trait_map_fns("gbif", cfg)
        if get_trait_number_from_id(fn.stem) in valid_traits
    ]
    splot_trait_map_fns = [
        fn
        for fn in get_trait_map_fns("splot", cfg)
        if get_trait_number_from_id(fn.stem) in valid_traits
    ]

    log.info("Combining sPlot and GBIF...")
    y_df = dd.concat(
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

    log.info("Closing Dask client...")
    close_dask(client, cluster)


if __name__ == "__main__":
    main()

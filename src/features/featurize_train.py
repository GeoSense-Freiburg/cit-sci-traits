"""Featurize training data."""

import argparse

import xarray as xr
from box import ConfigBox
from dask import config
from dask.distributed import Client

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import (
    compute_partitions,
    get_eo_fns_list,
    get_train_fn,
    get_trait_map_fns,
    group_y_fns,
    load_rasters_parallel,
    map_da_dtypes,
)


def cli() -> argparse.Namespace:
    """
    Parse command line arguments for featurizing training data.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Featurize training data")
    parser.add_argument(
        "-n",
        "--nchunks",
        type=int,
        default=9,
        help="Number of chunks to split data into",
    )
    return parser.parse_args()


def main(args: argparse.Namespace, cfg: ConfigBox = get_config()) -> None:
    """
    Main function for featurizing training data.

    Args:
        args (argparse.Namespace): Command-line arguments.
        cfg (ConfigBox): Configuration settings.

    Returns:
        None
    """
    system = cfg[cfg.system]

    with Client(
        dashboard_address=cfg.dask_dashboard,
        n_workers=system.featurize_train.n_workers,
        memory_limit=system.featurize_train.memory_limit,
    ), config.set({"array.slicing.split_large_chunks": False}):
        # Get trait map and EO data filenames
        trait_map_fns = get_trait_map_fns("interim")
        eo_fns = sorted(get_eo_fns_list("interim", datasets=cfg.datasets.X.keys()))

        log.info("Computing data types...")

        dtypes = map_da_dtypes(eo_fns + trait_map_fns, dask=True, nchunks=args.nchunks)

        log.info("Loading X data...")
        x_ds = load_rasters_parallel(eo_fns, nchunks=args.nchunks)

        log.info("Loading Y data...")
        trait_map_fns_grouped = group_y_fns(trait_map_fns)
        y_ds = load_rasters_parallel(
            trait_map_fns_grouped, cfg.datasets.Y.trait_stat, args.nchunks, ml_set="y"
        )
        y_source_ds = load_rasters_parallel(
            trait_map_fns_grouped,
            cfg.datasets.Y.trait_stat,
            args.nchunks,
            ml_set="y_source",
        )

        log.info("Combining data...")
        combined_ds = xr.merge([x_ds, y_ds, y_source_ds])

        # Get the names of the predictor and trait data_vars. traits start with "X{number}"
        # and predictors do not
        x_names = [dv for dv in combined_ds.data_vars if not str(dv).startswith("X")]
        y_names = [dv for dv in combined_ds.data_vars if str(dv).startswith("X")]

        # Change dtype of vodca variables to float32 since these will need to contain NaNs
        dtypes = {
            k: "float32" if k.startswith("vodca") else v for k, v in dtypes.items()
        }

        # Convert to Dask DataFrame and drop missing values
        ddf = (
            combined_ds.to_dask_dataframe()
            .drop(columns=["band", "spatial_ref"])
            .dropna(how="all", subset=y_names)
            .dropna(thresh=len(dtypes) // 7.5, subset=x_names)
        )

        log.info("Computing partitions...")
        df = compute_partitions(ddf).reset_index(drop=True)

    # Concatenate the chunks
    log.info("Writing to disk...")
    out_path = get_train_fn(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_path, compression="zstd", index=False)


if __name__ == "__main__":
    main(args=cli())

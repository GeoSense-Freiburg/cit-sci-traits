"""Featurize EO data for prediction and AoA calculation."""

import argparse
from pathlib import Path

from box import ConfigBox
from dask import config
import dask.dataframe as dd
from dask.distributed import Client
import xarray as xr

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import (
    compute_partitions,
    get_eo_fns_list,
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
        default=5,
        help="Number of chunks to split data into",
    )
    parser.add_argument("-m", "--memory-limit", type=str, default="100GB")
    parser.add_argument("-p", "--num-procs", type=int, default=None)
    return parser.parse_args()


def eo_ds_to_ddf(
    ds: xr.Dataset, dtypes: dict[str, str], sample: float = 1.0
) -> dd.DataFrame:
    """
    Convert an EO dataset to a Dask DataFrame.

    Parameters:
        ds (xr.Dataset): The input EO dataset.
        dtypes (dict[str, str]): A dictionary mapping variable names to their data types.

    Returns:
        dd.DataFrame: The converted Dask DataFrame.
    """

    # Change dtype of vodca variables to float32 since these will need to contain NaNs
    dtypes = {k: "float32" if k.startswith("vodca") else v for k, v in dtypes.items()}

    return (
        ds.to_dask_dataframe()
        .sample(frac=sample)
        .drop(columns=["band", "spatial_ref"])
        .dropna(thresh=len(dtypes) // 7.5, subset=list(dtypes.keys()))
    )


def main(args: argparse.Namespace, cfg: ConfigBox = get_config()) -> None:
    """Main function for featurizing EO data for prediction and AoA calculation."""
    with Client(
        dashboard_address=cfg.dask_dashboard,
        memory_limit=args.memory_limit,
        n_workers=args.num_procs,
    ), config.set({"array.slicing.split_large_chunks": False}):

        log.info("Getting filenames...")
        eo_fns = get_eo_fns_list(stage="interim")

        log.info("Mapping data types...")
        dtypes = map_da_dtypes(eo_fns, dask=True, nchunks=args.nchunks)

        log.info("Loading rasters...")
        ds = load_rasters_parallel(eo_fns, nchunks=args.nchunks)

        log.info("Converting to Dask DataFrame...")
        ddf = eo_ds_to_ddf(ds, dtypes)

        log.info("Computing partitions...")
        df = compute_partitions(ddf).reset_index(drop=True)

    out_path = (
        Path(cfg.train.dir)
        / cfg.eo_data.predict.dir
        / cfg.model_res
        / cfg.eo_data.predict.filename
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Saving DataFrame to %s...", out_path)
    df.to_parquet(out_path, compression="zstd", index=False, compression_level=19)

    log.info("Done!")


if __name__ == "__main__":
    main(cli())

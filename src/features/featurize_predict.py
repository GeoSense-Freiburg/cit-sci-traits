"""Featurize EO data for prediction and AoA calculation."""

import argparse
import math
import pickle

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from box import ConfigBox
from dask import config
from verstack import NaNImputer

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import (
    compute_partitions,
    get_eo_fns_list,
    get_predict_imputed_fn,
    get_predict_mask_fn,
    load_rasters_parallel,
)


def cli() -> argparse.Namespace:
    """Command line interface for featurizing EO data for prediction and AoA calculation."""
    parser = argparse.ArgumentParser(
        description="Featurize EO data for prediction and AoA calculation."
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    return parser.parse_args()


def impute_missing(df: pd.DataFrame, chunks: int | None = None) -> pd.DataFrame:
    """Impute missing values in a dataset using Verstack NaNImputer."""
    imputer = NaNImputer()
    if chunks is not None:
        df_chunks = np.array_split(df, chunks)
        df_imputed = pd.concat([imputer.impute(chunk) for chunk in df_chunks])
    else:
        df_imputed = imputer.impute(df)
    return df_imputed


def eo_ds_to_ddf(ds: xr.Dataset, thresh: float, sample: float = 1.0) -> dd.DataFrame:
    """
    Convert an EO dataset to a Dask DataFrame.

    Parameters:
        ds (xr.Dataset): The input EO dataset.
        dtypes (dict[str, str]): A dictionary mapping variable names to their data types.

    Returns:
        dd.DataFrame: The converted Dask DataFrame.
    """

    return (
        ds.to_dask_dataframe()
        .sample(frac=sample)
        .drop(columns=["band", "spatial_ref"])
        .dropna(
            thresh=math.ceil(len(ds.data_vars) * (1 - thresh)),
            subset=list(ds.data_vars),
        )
    )


def main(cfg: ConfigBox = get_config(), args: argparse.Namespace = cli()) -> None:
    """Main function for featurizing EO data for prediction and AoA calculation."""
    syscfg = cfg[detect_system()][cfg.model_res]["build_predict"]

    if args.debug:
        log.info("Running in debug mode...")

    log.info("Initializing Dask client...")
    client, cluster = init_dask(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.n_workers,
        memory_limit=syscfg.memory_limit,
    )
    config.set({"array.slicing.split_large_chunks": False})

    log.info("Getting filenames...")
    eo_fns = get_eo_fns_list(stage="interim")

    if args.debug:
        eo_fns = eo_fns[:2]

    log.info("Loading rasters...")
    ds = load_rasters_parallel(eo_fns, nchunks=syscfg.n_chunks)

    log.info("Converting to Dask DataFrame...")
    ddf = eo_ds_to_ddf(ds, thresh=cfg.train.missing_val_thresh)

    log.info("Computing partitions...")
    df = compute_partitions(ddf).reset_index(drop=True).set_index(["y", "x"])

    log.info("Closing Dask client...")
    close_dask(client, cluster)

    log.info("Creating mask for missing values...")
    mask = df.isna().reset_index(drop=False)

    mask_path = get_predict_mask_fn(cfg)

    if args.debug:
        mask = mask.head(5000)
        df = df.head(5000)
        mask_path = mask_path.parent / "debug" / mask_path.name

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving Mask to %s...", mask_path)
    mask.to_parquet(mask_path, compression="zstd", compression_level=19)

    log.info("Imputing missing values...")
    df_imputed = impute_missing(df, chunks=syscfg.impute_chunks)

    log.info("Casting dtypes of imputed data to conserve efficiency...")
    with open("reference/eo_data_dtypes.pkl", "rb") as f:
        dtypes = pickle.load(f)

    int_cols = [
        col for col, dtype in dtypes.items() if np.issubdtype(dtype, np.integer)
    ]

    # Round up the integer columns as imputing sometimes results in float values
    df_imputed[int_cols] = np.ceil(df_imputed[int_cols])
    df_imputed = df_imputed.astype(dtypes)

    log.info("Writing imputed predict DataFrame to disk...")
    pred_imputed_path = get_predict_imputed_fn(cfg)

    if args.debug:
        pred_imputed_path = pred_imputed_path.parent / "debug" / pred_imputed_path.name

    df_imputed.reset_index(drop=False).to_parquet(
        pred_imputed_path, compression="zstd", compression_level=19
    )

    log.info("Done!")


if __name__ == "__main__":
    main()

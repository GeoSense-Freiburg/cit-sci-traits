"""Featurize training data."""

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import xarray as xr
from box import ConfigBox
from dask import compute, config, delayed
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.raster_utils import open_raster


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


@delayed
def get_dtype_map(fn: Path, band: int, nchunks: int) -> Tuple[str, str]:
    """
    Get the data type map for a given file.

    Args:
        fn (Path): The file path.
        band (int): The band number.
        nchunks (int): The number of chunks.

    Returns:
        Tuple[str, str]: A tuple containing the long name and data type as strings.
    """
    band = 1
    da = open_raster(
        fn,
        chunks={"x": 36000 // nchunks, "y": 18000 // nchunks},
        mask_and_scale=False,
    )
    long_name = da.attrs["long_name"]

    if fn.stem[0] == "X":
        long_name = f"{fn.stem}_{long_name[band - 1]}"
        da.attrs["long_name"] = long_name

    dtype = da.sel(band=band).dtype
    da.close()

    return long_name, str(dtype)


@delayed
def load_x_raster(fn: Path, nchunks: int) -> Tuple[xr.DataArray, str]:
    """
    Load and preprocess an X raster file.

    Args:
        fn (Path): The file path of the raster file.
        nchunks (int): The number of chunks to divide the raster into.

    Returns:
        Tuple[xr.DataArray, str]: A tuple containing the loaded data array and the long
            name attribute.

    """
    da = open_raster(
        fn,
        chunks={"x": 36000 // nchunks, "y": 18000 // nchunks},
        mask_and_scale=True,
    ).sel(band=1)
    long_name = da.attrs["long_name"]

    return xr.DataArray(da), long_name


@delayed
def load_y_raster(
    trait_fns: List[Path], band: int, nchunks: int
) -> Tuple[xr.DataArray, str]:
    """
    Load and process raster data for a specific band.

    Args:
        trait_fns (List[Path]): List of file paths to the raster files.
        band (int): The band number to extract from the raster files.
        nchunks (int): Number of chunks to divide the raster into.

    Returns:
        Tuple[xr.DataArray, str]: A tuple containing the extracted band as a DataArray
        and the long name of the band.

    Raises:
        ValueError: If no files are found in `trait_fns`.
    """
    # find all matching files in fns
    if len(trait_fns) == 0:
        raise ValueError("No files found")

    das = []
    for raster_file in trait_fns:
        da = open_raster(
            raster_file,
            chunks={"x": 36000 // nchunks, "y": 18000 // nchunks},
            mask_and_scale=True,
        )

        long_name = da.attrs["long_name"]
        long_name = f"{raster_file.stem}_{long_name[band - 1]}"
        da.attrs["long_name"] = long_name

        das.append(da.sel(band=band))

    if len(das) == 1:
        return das[0], long_name

    # Find the array position of the fn in trait_fns that contains "gbif"
    gbif_idx = [i for i, fn in enumerate(trait_fns) if "gbif" in str(fn)][0]
    splot_idx = 1 - gbif_idx

    merged = xr.where(
        das[splot_idx].notnull(), das[splot_idx], das[gbif_idx], keep_attrs=True
    )

    for da in das:
        da.close()

    return merged, long_name


def main(args: argparse.Namespace, cfg: ConfigBox = get_config()) -> None:
    """
    Main function for featurizing training data.

    Args:
        args (argparse.Namespace): Command-line arguments.
        cfg (ConfigBox): Configuration settings.

    Returns:
        None
    """
    log.info("Initializing Dask...")
    cluster = LocalCluster(
        dashboard_address=":39143", n_workers=55, memory_limit="150GB"
    )
    client = Client(cluster)

    # Get trait map and EO data filenames
    trait_map_fns = []
    y_datasets = cfg.datasets.Y.use.split("_")
    for dataset in y_datasets:
        trait_maps_dir = (
            Path(cfg[dataset].interim.dir)
            / cfg[dataset].interim.traits
            / cfg.PFT
            / cfg.model_res
        )
        trait_map_fns += list(trait_maps_dir.glob("*.tif"))

    eo_data_dir = Path(cfg.eo_data.interim.dir) / cfg.model_res
    eo_fns = sorted(list(eo_data_dir.glob("*/*.tif")))

    # Sort trait_map_fns by number in file name (eg. X1, X2, X3)
    trait_map_fns = sorted(trait_map_fns, key=lambda x: int(x.stem.split("X")[-1]))

    log.info("Computing data types...")
    dtypes = set(
        compute(
            *[
                get_dtype_map(fn, cfg.datasets.Y.trait_stat, args.nchunks)
                for fn in eo_fns + trait_map_fns
            ]
        )
    )

    log.info("Loading X data...")
    x_arrays = list(compute(*[load_x_raster(fn, args.nchunks) for fn in eo_fns]))

    log.info("Loading Y data...")
    unique_traits = sorted(
        {fn.stem.split("_")[0] for fn in trait_map_fns},
        key=lambda x: int(x.split("X")[-1]),
    )
    trait_map_fns_grouped = [
        [fn for fn in trait_map_fns if trait == fn.stem] for trait in unique_traits
    ]
    y_arrays = list(
        compute(
            *[
                load_y_raster(trait_fns, cfg.datasets.Y.trait_stat, args.nchunks)
                for trait_fns in trait_map_fns_grouped
            ]
        )
    )

    log.info("Combining data...")
    all_arrays = x_arrays + y_arrays
    array_dict = {long_name: da for da, long_name in all_arrays}

    # Create dtype dictionary for type casting after dropping nans
    dtypes = dict(dtypes)

    # Create DataArray dict containing the name of each DA and the DA itself.
    # This will be used
    combined_ds = xr.Dataset(array_dict)

    # Get the names of the predictor and trait data_vars. traits start with "X{number}"
    # and predictors do not
    x_names = [dv for dv in combined_ds.data_vars if not dv.startswith("X")]
    y_names = [dv for dv in combined_ds.data_vars if dv.startswith("X")]

    with config.set({"array.slicing.split_large_chunks": False}):
        # Convert to Dask DataFrame and drop missing values
        ddf = (
            combined_ds.to_dask_dataframe()
            .drop(columns=["band", "spatial_ref"])
            .dropna(how="all", subset=y_names)
            .dropna(how="any", subset=x_names)
            .astype(dtypes)
        )

    try:
        log.info("Computing partitions...")
        npartitions = ddf.npartitions
        dfs = [
            ddf.get_partition(i).compute()
            for i in tqdm(
                range(npartitions), total=npartitions, desc="Computing partitions"
            )
        ]
    finally:
        client.close()
        cluster.close()

    # Concatenate the chunks
    log.info("Writing to disk...")
    out_dir = Path(cfg.train.dir) / cfg.PFT / cfg.model_res / cfg.datasets.Y.use
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.concat(dfs).to_parquet(
        out_dir / cfg.train.features, compression="zstd", index=False
    )


if __name__ == "__main__":
    main(args=cli())

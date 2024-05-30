"""Get the filenames of datasets based on the specified stage of processing."""

from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import xarray as xr
from box import ConfigBox
from dask import compute, delayed
from tqdm import trange

from src.conf.conf import get_config
from src.utils.raster_utils import open_raster


def get_eo_fns_dict(
    stage: str, datasets: str | list[str] | None = None
) -> dict[str, list[Path]]:
    """
    Get the filenames of EO datasets for a given stage.
    """
    cfg: ConfigBox = get_config()

    if isinstance(datasets, str):
        datasets = [datasets]

    stage_map = {
        "raw": {"path": Path(cfg.raw_dir), "ext": ".tif"},
        "interim": {
            "path": Path(cfg.interim_dir) / cfg.eo_data.interim.dir / cfg.model_res,
            "ext": ".tif",
        },
    }

    if stage not in stage_map:
        raise ValueError("Invalid stage. Must be one of 'raw', 'interim'.")

    fns = {}
    match stage:
        case "raw":
            for k, v in cfg.datasets.X.items():
                fns[k] = list(
                    stage_map[stage]["path"].glob(f"{v}/*{stage_map[stage]['ext']}")
                )

        case "interim":
            for k in cfg.datasets.X.keys():
                fns[k] = list(
                    stage_map[stage]["path"].glob(f"{k}/*{stage_map[stage]['ext']}")
                )

    if datasets is not None:
        fns = {k: v for k, v in fns.items() if k in datasets}

    return fns


def get_eo_fns_list(stage: str, datasets: str | list[str] | None = None) -> list[Path]:
    """
    Get the filenames of EO datasets for a given stage, flattened into a list.
    """
    fns = get_eo_fns_dict(stage, datasets)

    # Return flattened list of filenames
    return [fn for ds_fns in fns.values() for fn in ds_fns]


def map_da_dtype(fn: Path, band: int = 1, nchunks: int = 9) -> tuple[str, str]:
    """
    Get the data type map for a given file.

    Args:
        fn (Path): The file path.
        band (int): The band number.
        nchunks (int): The number of chunks.

    Returns:
        tuple[str, str]: A tuple containing the long name and data type as strings.
    """
    res = get_res(fn)

    data = open_raster(
        fn,
        chunks={"x": (360 / res) // nchunks, "y": (180 / res) // nchunks},
        mask_and_scale=False,
    )
    long_name: str = data.attrs["long_name"]

    if fn.stem[0] == "X":
        long_name = f"{fn.stem}_{long_name[band - 1]}"
        data.attrs["long_name"] = long_name
    else:
        band = 1  # Only traits have multiple bands

    dtype = str(data.sel(band=band).dtype)

    data.close()

    return long_name, dtype


def map_da_dtypes(
    fns: list[Path], band: int = 1, nchunks: int = 9, dask: bool = False
) -> dict[str, str]:
    """
    Map the data types of a list of files.

    Args:
        fns (list[Path]): A list of file paths.
        nchunks (int): The number of chunks.

    Returns:
        dict[str, str]: A dictionary mapping the long names to the data types.
    """
    if dask:
        dtypes = [delayed(map_da_dtype)(fn, band=band, nchunks=nchunks) for fn in fns]
        return dict(set(compute(*dtypes)))

    dtype_map: dict[str, str] = {}
    for fn in fns:
        long_name, dtype = map_da_dtype(fn, band=band, nchunks=nchunks)
        dtype_map[long_name] = dtype

    return dtype_map


def get_res(fn: Path) -> int | float:
    """
    Get the resolution of a raster.
    """
    data = open_raster(fn).sel(band=1)
    res = abs(data.rio.resolution()[0])
    data.close()
    del data
    return res


@delayed
def load_raster(fn: Path, nchunks: int) -> tuple[str, xr.DataArray]:
    """
    Load a raster dataset using delayed computation.

    Parameters:
        fn (Path): Path to the raster dataset file.
        nchunks (int): Number of chunks to divide the dataset into for parallel processing.

    Returns:
        tuple[xr.DataArray, str]: A tuple containing the loaded raster data as a DataArray
            and the long_name attribute of the dataset.

    Raises:
        ValueError: If multiple files are found while opening the raster dataset.
    """
    res = get_res(fn)
    da = open_raster(
        fn,
        chunks={"x": (360 / res) // nchunks, "y": (180 / res) // nchunks},
        mask_and_scale=True,
    ).sel(band=1)

    long_name = da.attrs["long_name"]

    return long_name, xr.DataArray(da)


def load_rasters_parallel(fns: list[Path], nchunks: int = 9) -> xr.Dataset:
    """
    Load multiple raster datasets in parallel using delayed computation.

    Parameters:
        fns (list[Path]): List of paths to the raster dataset files.
        nchunks (int): Number of chunks to divide each dataset into for parallel processing.

    Returns:
        dict[str, xr.DataArray]: A dictionary where keys are the long_name attributes of
            the datasets and values are the loaded raster data as DataArrays.
    """
    das: dict[str, xr.DataArray] = dict(
        compute(*[load_raster(fn, nchunks) for fn in fns])
    )
    return xr.Dataset(das)


def compute_partitions(ddf: dd.DataFrame) -> pd.DataFrame:
    """
    Compute the partitions of a Dask DataFrame and return the result as a Pandas DataFrame.

    Parameters:
        ddf (dd.DataFrame): The input Dask DataFrame.

    Returns:
        pd.DataFrame: The concatenated Pandas DataFrame containing all partitions of the
            input Dask DataFrame.
    """
    npartitions = ddf.npartitions
    dfs = [
        ddf.get_partition(i).compute()
        for i in trange(npartitions, desc="Computing partitions")
    ]
    return pd.concat(dfs)


if __name__ == "__main__":
    print(get_eo_fns_dict("interim"))

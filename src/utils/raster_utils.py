"""Utility functions for working with raster files."""

import gc
import multiprocessing
import os
from pathlib import Path
from typing import Any, Optional

import rasterio
import rioxarray as riox
import xarray as xr
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays, merge_datasets


def xr_to_raster(
    data: xr.DataArray | xr.Dataset,
    out: str | os.PathLike,
    dtype: Optional[Any] = None,
    compress: str = "ZSTD",
    num_threads: int = -1,
    **kwargs
) -> None:
    """Write a DataArray to a raster file."""
    dtype = dtype if dtype is not None else data.dtype
    if num_threads == -1:
        num_threads = multiprocessing.cpu_count()

    if Path(out).suffix == ".tif":
        tiff_opts = {
            "driver": "GTiff",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": compress,
            "num_threads": num_threads,
        }
        if dtype is not None:
            tiff_opts["dtype"] = dtype

        data.rio.to_raster(out, **tiff_opts, **kwargs)
    else:
        data.rio.to_raster(out, dtype=dtype, **kwargs)

    add_overviews(out)


def add_overviews(
    raster_file: str | os.PathLike, levels: Optional[list[int]] = None
) -> None:
    """Add overviews to a raster file."""
    if levels is None:
        levels = [2, 4, 8, 16, 32]

    with rasterio.open(raster_file, "r+") as raster:
        raster.build_overviews(levels, Resampling.average)
        raster.update_tags(ns="rio_overview", resampling="average")


def merge_rasters(
    raster_files: list[str | os.PathLike], out_file: str | os.PathLike
) -> None:
    """Merge a list of raster files into a single raster file.

    Args:
        raster_files (list[str]): A list of raster files to merge.
        out_file (str): The output file path.
    """
    rasters = [riox.open_rasterio(file) for file in raster_files]
    if isinstance(rasters[0], xr.DataArray):
        merged = merge_arrays(rasters)  # pyright: ignore[reportArgumentType]
    elif isinstance(rasters[0], xr.Dataset):
        merged = merge_datasets(rasters)  # pyright: ignore[reportArgumentType]
    elif isinstance(rasters[0], list):
        raise ValueError("Nested lists are not supported.")

    xr_to_raster(merged, out_file)

    merged.close()
    for raster in rasters:
        raster.close()

    del merged, rasters
    gc.collect()

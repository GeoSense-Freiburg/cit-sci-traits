"""Utility functions for working with raster files."""

import gc
import multiprocessing
import os
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
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


def open_raster(
    filename: str | os.PathLike, mask_and_scale: bool = True, **kwargs
) -> xr.DataArray | xr.Dataset:
    """Open a raster dataset using rioxarray."""
    ds = riox.open_rasterio(filename, mask_and_scale=mask_and_scale, **kwargs)

    if isinstance(ds, list):
        raise ValueError("Multiple files found.")

    return ds


def create_sample_raster(
    extent: list[int] | list[float] | None = None, resolution: int | float = 1
) -> xr.DataArray:
    """Sample a raster at a given resolution."""
    if extent is None:
        extent = [-180, -90, 180, 90]

    xmin, ymin, xmax, ymax = extent
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
    half_res = resolution * 0.5

    x_data = np.linspace(xmin + half_res, xmax - half_res, width, dtype=np.float64)
    y_data = np.linspace(ymax - half_res, ymin + half_res, height, dtype=np.float64)
    x_var = xr.DataArray(x_data, dims="x", attrs={"units": "degrees_east"})
    y_var = xr.DataArray(y_data, dims="y", attrs={"units": "degrees_north"})
    coords = {"x": x_var, "y": y_var}

    data = np.zeros((height, width), dtype=np.uint8)
    data_array = xr.DataArray(data, dims=("y", "x"), coords=coords)

    return data_array.rio.write_crs("EPSG:4326")


def raster_to_df(
    rast: xr.DataArray, rast_name: str | None = None, gdf: bool = False
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Convert a raster to a DataFrame or GeoDataFrame.

    Parameters:
        rast (xr.DataArray): The input raster data.
        rast_name (str, optional): The name of the raster. If not provided, it will be
            extracted from the 'long_name' attribute of the raster.
        gdf (bool, optional): If True, return a GeoDataFrame instead of a DataFrame.

    Returns:
        pd.DataFrame | gpd.GeoDataFrame: The converted DataFrame or GeoDataFrame.

    Raises:
        ValueError: If the raster does not have a 'long_name' attribute.

    Notes:
        - The function converts the raster data to a DataFrame or GeoDataFrame.
        - If `gdf` is True, the function returns a GeoDataFrame with geometry based on the
          x and y coordinates of the DataFrame.

    """
    if rast_name is None:
        if "long_name" not in rast.attrs:
            raise ValueError("Raster must have a 'long_name' attribute.")

        rast_name = rast.attrs["long_name"]

    # Get coordinate names that aren't x or y
    coords = [coord for coord in rast.coords if coord not in ["x", "y"]]

    df = (
        rast.to_dataframe(rast_name)
        .drop(columns=coords)
        .reset_index()
        .dropna(ignore_index=True)
    )

    if gdf:
        return gpd.GeoDataFrame(
            df[df.columns.difference(rast.dims)],
            geometry=gpd.points_from_xy(df.x, df.y),
            crs=rast.rio.crs,
        )

    return df

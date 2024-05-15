"""Utility functions for working with raster files."""

import gc
import multiprocessing
import os
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as riox
import xarray as xr
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays, merge_datasets


def encode_nodata(da: xr.DataArray, dtype: str | np.dtype) -> xr.DataArray:
    """Encode the nodata value of a DataArray."""
    nodata = (
        np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else np.finfo(dtype).max
    )
    if da.max() == nodata and not da.max() == da.rio.nodata:
        # use min_val as nodata
        nodata = (
            np.iinfo(dtype).min
            if np.issubdtype(dtype, np.integer)
            else np.finfo(dtype).min
        )

    if dtype != da.dtype:
        da = da.fillna(nodata)
        da = da.astype(dtype)

    if np.issubdtype(dtype, np.integer):
        return da.rio.write_nodata(nodata)

    return da.rio.write_nodata(nodata, encoded=True)


def scale_data(
    da: xr.DataArray, dtype: str | np.dtype, all_pos: bool = False
) -> xr.DataArray:
    if all_pos:
        return (da - da.min()) * (np.iinfo(dtype).max - 1) / (da.max() - da.min())

    return (da - da.min()) * (np.iinfo(dtype).max - np.iinfo(dtype).min - 1) / (
        da.max() - da.min()
    ) + np.iinfo(dtype).min


def xr_to_raster(
    data: xr.DataArray | xr.Dataset,
    out: str | os.PathLike,
    dtype: np.dtype | str | None = None,
    compress: str = "ZSTD",
    num_threads: int = -1,
    **kwargs
) -> None:
    """Write a DataArray to a raster file."""
    if isinstance(data, xr.DataArray):
        dtype = dtype if dtype is not None else data.dtype
        data = encode_nodata(data, dtype)
    else:
        dtype = dtype if dtype is not None else data[list(data.data_vars)[0]].dtype
        for dv in data.data_vars:
            data[dv] = encode_nodata(data[dv], dtype)

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
) -> xr.Dataset:
    """Generate a sample raster at a given resolution."""
    if extent is None:
        extent = [-180, -90, 180, 90]

    xmin, ymin, xmax, ymax = extent
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
    half_res = resolution * 0.5
    decimals = int(np.ceil(-np.log10(half_res)))

    x_data = np.round(
        np.linspace(xmin + half_res, xmax - half_res, width, dtype=np.float64), decimals
    )
    y_data = np.round(
        np.linspace(ymax - half_res, ymin + half_res, height, dtype=np.float64),
        decimals,
    )

    ds = xr.Dataset({"y": (("y"), y_data), "x": (("x"), x_data)})

    return ds.rio.write_crs("EPSG:4326")


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

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
from ease_grid import EASE2_grid
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays, merge_datasets

from src.conf.environment import log


def set_nodata(da: xr.DataArray, dtype: str | np.dtype) -> xr.DataArray:
    """Encode the nodata value of a DataArray."""
    da = da.copy()
    nodata = np.iinfo(dtype).min if np.issubdtype(dtype, np.integer) else np.nan

    if da.rio.nodata == nodata and da.rio.encoded_nodata is None:
        return da

    if da.rio.nodata is not None:
        if np.isnan(da.rio.nodata) and np.isnan(nodata):
            return da

        if np.isnan(nodata) and not np.isnan(da.rio.nodata):
            da = da.rio.write_nodata(nodata, encoded=False)
            return da

    # If the da nodata value is set to nan but the data should be of integer type
    # Fill the nans with the nodata value and set the data type
    if da.isnull().any():
        da = da.fillna(nodata)
    da = da.astype(dtype)
    da = da.rio.write_nodata(nodata, encoded=False)
    return da


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
    encode_nodata: bool = True,
    **kwargs: dict[str, Any],
) -> None:
    """Write a DataArray to a raster file."""
    if encode_nodata:
        if isinstance(data, xr.DataArray):
            dtype = dtype if dtype is not None else data.dtype
            data = set_nodata(data, dtype)
        else:
            dtype = dtype if dtype is not None else data[list(data.data_vars)[0]].dtype
            for dv in data.data_vars:
                data[dv] = set_nodata(data[dv], dtype)

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
        levels = [2, 4, 8, 16]

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
    else:
        raise ValueError("Raster type not recognized.")

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


def coord_decimal_places(resolution: int | float):
    """Returns the number of decimal places needed to express the centroid of a grid cell
    at the target resolution"""
    centroid_step = resolution / 2

    result_str = f"{centroid_step:.20f}".rstrip("0")  # Keep only significant digits
    if "." in result_str:
        decimal_part = result_str.split(".")[1]
        return len(decimal_part)

    return 0


def generate_epsg4326_grid(
    resolution: int | float, extent: list[int | float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a grid of x and y coordinates in EPSG:4326."""
    if extent is None:
        extent = [-180, -90, 180, 90]

    xmin, ymin, xmax, ymax = extent
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
    half_res = resolution * 0.5
    decimals = coord_decimal_places(resolution)

    x_coords = np.round(
        np.linspace(xmin + half_res, xmax - half_res, width, dtype=np.float64),
        decimals,
    )
    y_coords = np.round(
        np.linspace(ymax - half_res, ymin + half_res, height, dtype=np.float64),
        decimals,
    )

    return x_coords, y_coords


def generate_epsg6933_grid(
    resolution: int | float, extent: list[int | float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a grid of x and y coordinates in EPSG:6933."""
    ease_grid = EASE2_grid(res=resolution)

    if extent is None:
        extent = [
            ease_grid.x_min,
            ease_grid.y_min,
            ease_grid.x_max,
            ease_grid.y_max,
        ]

    xmin, ymin, xmax, ymax = extent
    half_res_x = ease_grid.x_pixel / 2
    half_res_y = ease_grid.y_pixel / 2
    decimals_x = coord_decimal_places(ease_grid.x_pixel)
    decimals_y = coord_decimal_places(ease_grid.y_pixel)

    x_coords = np.round(
        np.linspace(
            xmin + half_res_x,
            xmax - half_res_x,
            ease_grid.shape[1],
        ),
        decimals_x,
    )
    y_coords = np.round(
        np.linspace(
            ymax - half_res_y,
            ymin + half_res_y,
            ease_grid.shape[0],
        ),
        decimals_y,
    )

    return x_coords, y_coords


def create_sample_raster(
    extent: list[int | float] | None = None,
    resolution: int | float = 1,
    crs: str = "EPSG:4326",
) -> xr.Dataset:
    """
    Generate a sample raster at a given resolution and CRS.

    Parameters:
    extent (list[int | float] | None): The spatial extent of the raster in the format
        [min_x, min_y, max_x, max_y]. If None, a default extent will be used based on
        the CRS.
    resolution (int | float): The resolution of the raster grid cells.
    crs (str): The coordinate reference system (CRS) of the raster. Default is "EPSG:4326".

    Returns:
    xr.Dataset: An xarray Dataset containing the generated raster with the specified CRS.

    Raises:
    ValueError: If the CRS is not "EPSG:4326" or "EPSG:6933" and extent is not provided.
    """

    if crs == "EPSG:4326":
        x_coords, y_coords = generate_epsg4326_grid(resolution, extent)
    elif crs == "EPSG:6933":
        x_coords, y_coords = generate_epsg6933_grid(resolution, extent)
    else:
        raise ValueError("Extent must be provided for non-EPSG:4326 CRS.")

    ds = xr.Dataset({"y": (("y"), y_coords), "x": (("x"), x_coords)})

    return ds.rio.write_crs(crs)


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


def pack_data(
    data: np.ndarray, nbits: int = 16, signed: bool = True
) -> tuple[np.float64, np.float64, int, np.ndarray]:
    """
    Packs the given data into a specified integer format with optional scaling and offset.
    Parameters:
    -----------
    data : np.ndarray
        The input data array to be packed.
    nbits : int, optional
        The number of bits for the integer representation (default is 16).
    signed : bool, optional
        Whether to use a signed integer type (default is True).
    Returns:
    --------
    tuple[np.float64, np.float64, int, np.ndarray]
        A tuple containing:
        - scale (np.float64): The scale factor used for packing.
        - offset (np.float64): The offset value used for packing.
        - nodata_value (int): The value used to represent missing data.
        - data_int16 (np.ndarray): The packed data array in the specified integer format.
    Raises:
    -------
    FloatingPointError
        If there is an invalid floating point operation during the packing process.
    Notes:
    ------
    This function scales and offsets the input data to fit into the specified integer format.
    It handles NaN values by replacing them with a designated nodata value.
    """

    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    dtype = f"int{nbits}" if signed else f"uint{nbits}"

    nodata_value: int = -(2 ** (nbits - 1)) if signed else 0
    scale = (data_max - data_min) / (2**nbits - 2)
    scale = np.float64(scale)
    offset = (data_max + data_min) / 2 if signed else data_min - scale
    offset = np.float64(offset)

    np.seterr(invalid="raise")
    try:
        data_int16 = np.where(
            np.isnan(data), nodata_value, np.round((data - offset) / scale)
        ).astype(dtype)
    except FloatingPointError as e:
        log.error("FloatingPointError: %s", e)
        log.error("data_min: %s", data_min)
        log.error("data_max: %s", data_max)
        log.error("scale: %s", scale)
        log.error("offset: %s", offset)

        offset_data = np.subtract(data, offset)
        log.error("offset_data stats:")
        log.error("min: %s", np.nanmin(offset_data))
        log.error("max: %s", np.nanmax(offset_data))

        scaled_data = np.divide(offset_data, scale)
        log.error("scaled data stats:")
        log.error("min: %s", np.nanmin(scaled_data))
        log.error("max: %s", np.nanmax(scaled_data))

        rounded_data = scaled_data.round()
        log.error("rounded_data stats:")
        log.error("min: %s", np.nanmin(rounded_data))
        log.error("max: %s", np.nanmax(rounded_data))

        cast_data = rounded_data.astype(dtype)
        log.error("cast_data stats:")
        log.error("min: %s", np.nanmin(cast_data))
        log.error("max: %s", np.nanmax(cast_data))
        raise

    return scale, offset, nodata_value, data_int16


def pack_xr(
    data: xr.DataArray | xr.Dataset, nbits: int = 16, signed: bool = True
) -> xr.DataArray | xr.Dataset:
    """
    Pack the given xarray DataArray or Dataset into a specified integer format with optional scaling and offset.
    Parameters:
    -----------
    data : xr.DataArray | xr.Dataset
        The input xarray DataArray or Dataset to be packed.
    nbits : int, optional
        The number of bits for the integer representation (default is 16).
    signed : bool, optional
        Whether to use a signed integer type (default is True).
    Returns:
    --------
    xr.DataArray | xr.Dataset
        The packed xarray DataArray or Dataset in the specified integer format.
    Raises:
    -------
    FloatingPointError
        If there is an invalid floating point operation during the packing process.
    Notes:
    ------
    This function scales and offsets the input data to fit into the specified integer format.
    It handles NaN values by replacing them with a designated nodata value.
    """
    data = data.copy()

    if isinstance(data, xr.DataArray):
        scale, offset, nodata_value, data_int16 = pack_data(data.values, nbits, signed)
        data.values = data_int16
        data.attrs["scale_factor"] = scale
        data.attrs["add_offset"] = offset
        data = data.rio.write_nodata(nodata_value, encoded=False)
        return data

    for var in data.data_vars:
        scale, offset, nodata_value, data_int16 = pack_data(
            data[var].values, nbits, signed
        )
        data[var].values = data_int16
        data[var].attrs["scale_factor"] = scale
        data[var].attrs["add_offset"] = offset
        data[var] = data[var].rio.write_nodata(nodata_value, encoded=False)

    return data

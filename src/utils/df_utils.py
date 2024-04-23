"""Utility functions for working with DataFrames and GeoDataFrames."""

from pathlib import Path

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd


def write_dgdf_parquet(dgdf: dgpd.GeoDataFrame, out_path: str | Path, **kwargs) -> None:
    """Write a Dask GeoDataFrame to a Parquet file."""
    if "write_index" not in kwargs:
        kwargs["write_index"] = False
    if "overwrite" not in kwargs:
        kwargs["overwrite"] = True
    dgdf.to_parquet(out_path, **kwargs)


def write_ddf_parquet(ddf: dd.DataFrame, out_path: str | Path, **kwargs) -> None:
    """Write a Dask DataFrame to a Parquet file."""
    if "write_index" not in kwargs:
        kwargs["write_index"] = False
    if "overwrite" not in kwargs:
        kwargs["overwrite"] = True
    ddf.to_parquet(out_path, **kwargs)


def write_df(
    df: pd.DataFrame | gpd.GeoDataFrame,
    out_path: str | Path,
    writer: str = "parquet",
    dask: bool = False,
    **kwargs,
) -> None:
    """Write a DataFrame to a file."""
    if writer == "parquet":
        if "compression" not in kwargs:
            kwargs["compression"] = "zstd"
        if "engine" not in kwargs:
            kwargs["engine"] = "pyarrow"

        if dask:
            npartitions = 64
            if isinstance(df, gpd.GeoDataFrame):
                dgdf = dgpd.from_geopandas(df, npartitions=npartitions)
                write_dgdf_parquet(dgdf, out_path, **kwargs)
            elif isinstance(df, pd.DataFrame):
                ddf = dd.from_pandas(df, npartitions=npartitions)
                write_ddf_parquet(ddf, out_path, **kwargs)
        else:
            df.to_parquet(out_path, index=False, **kwargs)
    else:
        raise ValueError("Invalid writer.")


def optimize_column(col: pd.Series) -> pd.Series:
    """
    Optimize the data type of a column in a DataFrame or GeoDataFrame.

    Parameters:
        col (pd.Series): The input column to optimize.

    Returns:
        pd.Series: The optimized column.

    """
    min_val, max_val = col.min(), col.max()
    if col.dtype in [np.int64, np.int32, np.int16]:
        if min_val > np.iinfo(np.int8).min and max_val < np.iinfo(np.int8).max:
            col = col.astype(np.int8)
        elif min_val > np.iinfo(np.int16).min and max_val < np.iinfo(np.int16).max:
            col = col.astype(np.int16)
        elif min_val > np.iinfo(np.int32).min and max_val < np.iinfo(np.int32).max:
            col = col.astype(np.int32)
    elif col.dtype in [np.float64, np.float32]:
        if min_val > np.finfo(np.float16).min and max_val < np.finfo(np.float16).max:
            col_temp = col.astype(np.float16)
            if not ((col - col_temp).abs() > 0.001).any():
                col = col_temp
        elif min_val > np.finfo(np.float32).min and max_val < np.finfo(np.float32).max:
            col_temp = col.astype(np.float32)
            if not ((col - col_temp).abs() > 1e-6).any():
                col = col_temp

        # Check if all float values are actually integers
        if (col % 1 == 0).all():
            col = col.astype(np.int64)
            # Recursively call optimize_column to further optimize the integer column
            col = optimize_column(col)
    return col


def optimize_columns(
    df: pd.DataFrame | gpd.GeoDataFrame,
    coords_as_categories: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Optimize the columns of a DataFrame or GeoDataFrame.

    This function iterates over each column of the input DataFrame or GeoDataFrame
    and optimizes the columns that are of type pd.Series. The optimization is done
    by calling the `optimize_column` function on each column.

    Parameters:
        df (pd.DataFrame | gpd.GeoDataFrame): The input DataFrame or GeoDataFrame.

    Returns:
        pd.DataFrame | gpd.GeoDataFrame: The optimized DataFrame or GeoDataFrame.
    """
    for column in df.columns:
        col = df[column]
        if isinstance(col, gpd.GeoSeries):
            continue
        if isinstance(col, pd.Series):
            if column in ["x", "y"] and coords_as_categories:
                df[column] = col.astype("category")
            else:
                df[column] = optimize_column(col)
    return df

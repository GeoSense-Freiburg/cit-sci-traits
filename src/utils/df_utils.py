"""Utility functions for working with DataFrames and GeoDataFrames."""

from pathlib import Path

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd

from src.utils.log_utils import setup_logger

log = setup_logger(__name__, "INFO")


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


def optimize_column(col: pd.Series, categorize: bool = False) -> pd.Series:
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
        # TODO: Implement float optimization that considers the precision loss

        # if min_val > np.finfo(np.float32).min and max_val < np.finfo(np.float32).max:
        #     col_temp = col.astype(np.float32)
        #     if not ((col - col_temp).abs() > 1e-6).any():
        #         col = col_temp
        # elif min_val > np.finfo(np.float16).min and max_val < np.finfo(np.float16).max:
        #     col_temp = col.astype(np.float16)
        #     if not ((col - col_temp).abs() > 0.001).any():
        #         col = col_temp

        # Check if all float values are actually integers
        if (col % 1 == 0).all():
            col = col.astype(np.int64)
            # Recursively call optimize_column to further optimize the integer column
            col = optimize_column(col)
    elif col.dtype == "object":
        col = col.astype("string[pyarrow]")

    if categorize and len(col.unique()) < 0.5 * len(col):
        col = col.astype("category")
    return col


def optimize_columns(
    df: pd.DataFrame | gpd.GeoDataFrame,
    coords_as_categories: bool = False,
    categorize: bool = False,
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
                df[column] = optimize_column(col, categorize=categorize)
    return df


def outlier_mask(
    col: pd.Series, lower: float = 0.05, upper: float = 0.95
) -> np.ndarray:
    """
    Returns a boolean mask indicating whether each value in the input column is an outlier or not.

    Parameters:
        col (pd.Series): The input column.
        lower (float): The lower quantile threshold for determining outliers.
            Defaults to 0.05.
        upper (float): The upper quantile threshold for determining outliers.
            Defaults to 0.95.

    Returns:
        np.ndarray: A boolean mask indicating whether each value is an outlier or not.
    """
    col_values = col.values
    lower_bound, upper_bound = np.quantile(
        col_values, [lower, upper]  # pyright: ignore[reportArgumentType]
    )
    return (col_values >= lower_bound) & (col_values <= upper_bound)


def filter_outliers(
    df: pd.DataFrame,
    cols: list[str],
    quantiles: tuple[float, float] = (0.05, 0.95),
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Filter outliers from the input DataFrame.

    This function filters outliers from the input DataFrame by applying the `outlier_mask`
    function on each column in the input DataFrame that is specified in the `cols` list.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cols (list[str]): The list of column names to filter outliers from.
        quantiles (tuple[float]): A tuple of two floats representing the lower and upper
            quantiles for outlier detection.

    Returns:
        pd.DataFrame: The DataFrame with outliers filtered out.
    """
    if not set(cols).issubset(df.columns):
        raise ValueError("Columns not found in DataFrame.")
    masks = [outlier_mask(df[col], *quantiles) for col in cols]
    comb_mask = np.all(masks, axis=0)

    if verbose:
        num_dropped = len(df) - comb_mask.sum()
        pct_dropped = (num_dropped / len(df)) * 100
        log.info("Dropping %d rows (%.2f%%)", num_dropped, pct_dropped)
    return df[comb_mask]

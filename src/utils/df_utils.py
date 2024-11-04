"""Utility functions for working with DataFrames and GeoDataFrames."""

import gc
import warnings
from pathlib import Path
from typing import Any

import dask
import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from affine import Affine

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.log_utils import setup_logger
from src.utils.raster_utils import create_sample_raster, xr_to_raster

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
    use_dask: bool = False,
    **kwargs,
) -> None:
    """Write a DataFrame to a file."""
    if writer == "parquet":
        if "compression" not in kwargs:
            kwargs["compression"] = "zstd"
        if "engine" not in kwargs:
            kwargs["engine"] = "pyarrow"

        if use_dask:
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
    lower_bound, upper_bound = np.quantile(  # pyright: ignore[reportCallIssue]
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


def reproject_geo_to_xy(
    df: pd.DataFrame, crs: str = "EPSG:6933", x: str = "x", y: str = "y"
):
    """
    Reprojects geographical coordinates to a specified coordinate reference system (CRS).

    Parameters:
    df (pd.DataFrame): DataFrame containing the geographical coordinates.
    crs (str): Coordinate reference system to reproject to. Default is "EPSG:6933".
    x (str): Name of the column containing the x (longitude) coordinates. Default is "x".
    y (str): Name of the column containing the y (latitude) coordinates. Default is "y".

    Returns:
    pd.DataFrame: DataFrame with reprojected x and y coordinates.
    """
    proj = pyproj.Proj(pyproj.CRS(crs))
    xy = proj(df[x], df[y])
    df["x"] = xy[0]
    df["y"] = xy[1]
    return df


def point_to_cell_index(
    x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, transform: Affine
) -> tuple[int, int]:
    """
    Converts point coordinates to cell indices based on a given affine transform.

    Args:
        x (pd.Series | np.ndarray): The x-coordinates of the points.
        y (pd.Series | np.ndarray): The y-coordinates of the points.
        transform (Affine): The affine transformation to apply.

    Returns:
        tuple[int, int]: The column and row indices corresponding to the input points.
    """
    cols, rows = ~transform * (x, y)  # pyright: ignore[reportOperatorIssue]
    return cols.astype(int), rows.astype(int)


def xy_to_rowcol_df(
    df: pd.DataFrame,
    transform: Affine | tuple[float, float, float, float, float, float],
    x: str = "x",
    y: str = "y",
) -> pd.DataFrame:
    """
    Converts x, y coordinates in a DataFrame to row, column indices based on a given
    affine transform.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing x, y coordinates.
    transform : Affine or tuple
        The affine transformation or a tuple representing the affine transformation
        parameters.
    x : str, optional
        The column name for x coordinates in the DataFrame. Default is "x".
    y : str, optional
        The column name for y coordinates in the DataFrame. Default is "y".

    Returns:
    --------
    pd.DataFrame
        A DataFrame with additional columns 'col' and 'row' representing the column and
        row indices.
    """
    df = df.copy()
    if isinstance(transform, tuple):
        transform = Affine.from_gdal(*transform)

    idx = point_to_cell_index(df[x], df[y], transform)
    df["col"] = idx[0]
    df["row"] = idx[1]
    return df


def agg_df(
    df: pd.DataFrame,
    by: str | list[str],
    data: str,
    funcs: dict[str, Any] | None = None,
    n_min: int = 1,
    n_max: int | None = None,
) -> pd.DataFrame:
    """
    Aggregate a DataFrame by specified columns and apply aggregation functions.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to be aggregated.
    by : str or list of str
        Column(s) to group by.
    data : str
        The column to aggregate.
    funcs : dict of str to Any, optional
        A dictionary where keys are the names of the resulting columns and values are
        the aggregation functions.
        If None, the default aggregation functions are used: mean, std, median, q05, q95,
        count.
    n_min : int, optional
        Minimum count threshold to filter groups. Default is 1.
    n_max : int, optional
        Maximum count threshold to filter groups. Default is None.

    Returns:
    --------
    pd.DataFrame
        The aggregated DataFrame with the specified aggregation functions applied.
    """
    if funcs is None:
        funcs = {
            "mean": "mean",
            "std": "std",
            "median": "median",
            "q05": lambda x: x.quantile(0.05, interpolation="nearest"),
            "q95": lambda x: x.quantile(0.95, interpolation="nearest"),
            "count": "count",
        }

    if n_max is not None:
        # Randomly subsample a maximum of n_max points from each group
        seed = get_config().random_seed
        df = df.groupby(by, observed=False, group_keys=False).apply(
            lambda x: x.sample(n=min(n_max, len(x)), random_state=seed)
        )

    df = df.groupby(by, observed=False)[[data]].agg(list(funcs.values()))

    df.columns = list(funcs.keys())

    if "count" in df.columns:
        if n_min > 1:
            df = df[df["count"] >= n_min]

    return df.reset_index()


def rasterize_points(
    df: pd.DataFrame,
    data: str,
    x: str = "x",
    y: str = "y",
    raster: xr.DataArray | xr.Dataset | None = None,
    res: int | float | None = None,
    crs: str | None = None,
    nodata: int | float = np.nan,
    funcs: dict[str, Any] | None = None,
    n_min: int = 1,
    n_max: int | None = None,
) -> xr.Dataset:
    """
    Rasterizes point data from a DataFrame into a raster dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the point data to be rasterized.
    data : str
        Column name in the DataFrame containing the data values to be rasterized.
    x : str, optional
        Column name for the x-coordinates, by default "x".
    y : str, optional
        Column name for the y-coordinates, by default "y".
    raster : xr.DataArray or xr.Dataset, optional
        Existing raster to use as a reference. If None, a new raster will be created.
    res : int or float, optional
        Resolution of the new raster if `raster` is None.
    crs : str, optional
        Coordinate reference system of the new raster if `raster` is None.
    nodata : int or float, optional
        Value to use for no-data cells, by default np.nan.
    funcs : dict[str, Any], optional
        Aggregation functions to apply to the data, by default None.
    n_min : int, optional
        Minimum number of points required to aggregate, by default 1.
    n_max : int, optional
        Maximum number of points to aggregate, by default None.

    Returns:
    --------
    xr.Dataset
        Raster dataset with the rasterized point data.

    Raises:
    -------
    ValueError
        If neither `raster` nor both `res` and `crs` are provided.
        If `res` and `crs` are provided when `raster` is also provided.
    """
    if raster is None:
        if res is None or crs is None:
            raise ValueError(
                "Either 'raster' or both 'res' and 'crs' must be provided."
            )
        # Generate a sample raster with the specified resolution and CRS
        ref = create_sample_raster(resolution=res, crs=crs)

    elif res is not None or crs is not None:
        raise ValueError(
            "'res' and 'crs' must not be provided if 'raster' is provided."
        )
    else:
        # Create an empty raster with the same shape, CRS, resolution, and nodata value
        ref = raster.copy()
        # Drop all existing data variables
        for var in ref.data_vars:
            ref = ref.drop_vars(var)  # pyright: ignore[reportArgumentType]

    transform = ref.rio.transform().to_gdal()

    grid_df = (
        xy_to_rowcol_df(df, transform, x=x, y=y)
        .drop(columns=["x", "y"])
        .pipe(
            agg_df, by=["row", "col"], data=data, funcs=funcs, n_min=n_min, n_max=n_max
        )
    )

    if dask.is_dask_collection(grid_df):
        grid_df = grid_df.compute()  # pyright: ignore[reportCallIssue]

    # Write each column of the DataFrame to a separate data variable in the raster
    for col in grid_df.columns:
        if str(col) in ("row", "col"):
            continue
        ref[col] = (("y", "x"), np.full(ref.rio.shape, nodata))
        ref[col].values[grid_df["row"].values, grid_df["col"].values] = grid_df[
            col
        ].values

        if np.isnan(nodata):
            ref[col] = ref[col].rio.write_nodata(nodata, encoded=False)
        else:
            ref[col] = ref[col].rio.write_nodata(nodata, encoded=True)

    return ref  # pyright: ignore[reportReturnType]


def global_grid_df(
    df: pd.DataFrame,
    col: str,
    lon: str = "decimallongitude",
    lat: str = "decimallatitude",
    res: int | float = 0.5,
    stats: list | None = None,
    n_min: int = 1,
) -> pd.DataFrame:
    """
    Calculate gridded statistics for a given DataFrame.

    Args:
        df (dd.DataFrame): The input DataFrame.
        col (str): The column name for which to calculate statistics.
        lon (str, optional): The column name for longitude values. Defaults to "decimallongitude".
        lat (str, optional): The column name for latitude values. Defaults to "decimallatitude".
        res (int | float, optional): The resolution of the grid. Defaults to 0.5.
        stats (list, optional): The list of statistics to calculate. Defaults to None.
    Returns:
        pd.DataFrame: A DataFrame containing gridded statistics.

    """
    warnings.warn(
        "'global_grid_df' is deprecated and will be removed in a future "
        "version. Use 'rasterize_points' instead.",
        DeprecationWarning,
    )

    stat_funcs = {
        "mean": "mean",
        "std": "std",
        "median": "median",
        "q05": lambda x: x.quantile(0.05, interpolation="nearest"),
        "q95": lambda x: x.quantile(0.95, interpolation="nearest"),
        "count": "count",
    }

    # if stats is not None:
    #     stat_funcs = {k: v for k, v in stat_funcs.items() if k in stats}

    # Calculate the bin for each row directly
    df = df.copy()
    # if the copy warning still persists, set y and x with df.loc[:, "y"], etc.
    df["y"] = (df[lat] + 90) // res * res - 90 + res / 2
    df["x"] = (df[lon] + 180) // res * res - 180 + res / 2

    gridded_df = (
        df.drop(columns=[lat, lon])
        .groupby(["y", "x"], observed=False)[[col]]
        .agg(list(stat_funcs.values()))
    )

    gridded_df.columns = list(stat_funcs.keys())

    if n_min > 1:
        gridded_df = gridded_df[gridded_df["count"] >= n_min]

    if stats is not None:
        return gridded_df[stats]

    return gridded_df


def grid_df_to_raster(
    df: pd.DataFrame,
    res: int | float,
    out: Path | None = None,
    name: str = "trait",
    *args: Any,
    **kwargs: Any,
) -> None | xr.Dataset:
    """
    Converts a grid DataFrame to a raster file.

    Args:
        df (pd.DataFrame): The grid DataFrame to convert.
        res (int | float): The resolution of the raster.
        out (Path): The output path for the raster file.

    Returns:
        None
    """
    ref = create_sample_raster(resolution=res)
    decimals = coord_decimal_places(res)

    ds = (
        df.to_xarray()
        .assign_coords(
            x=lambda ds: np.round(ds["x"], decimals),
            y=lambda ds: np.round(ds["y"], decimals),
        )
        .merge(ref, join="right")
        .rio.write_crs(ref.rio.crs)
    )

    for var in ds.data_vars:
        if "_FillValue" in ds[var].attrs:
            nodata = ds[var].attrs["_FillValue"]
            ds[var] = ds[var].where(ds[var] != nodata, np.nan)
        else:
            ds[var] = ds[var].where(~ds[var].isnull(), np.nan)

        ds[var] = ds[var].rio.write_nodata(-32767.0, encoded=True)

    ds.attrs["long_name"] = list(ds.data_vars)
    if out is None:
        ds.attrs["trait"] = name
    else:
        ds.attrs["trait"] = out.stem

    if out is not None:
        xr_to_raster(ds, out, *args, **kwargs)

    ref.close()

    if out is None:
        return ds

    ds.close()

    del ref, ds
    gc.collect()
    return None


def pipe_log(df: pd.DataFrame, message: str) -> pd.DataFrame:
    """Simple function to log a message during method chaining with a DataFrame."""
    log.info(message)
    return df

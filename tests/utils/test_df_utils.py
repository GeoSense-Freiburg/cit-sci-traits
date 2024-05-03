"""Tests for the df_utils module."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from src.utils.df_utils import optimize_column, optimize_columns

# pylint: disable=missing-function-docstring


@pytest.fixture(name="sample_dataframe")
def fixt_sample_dataframe():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0],
            "C": ["a", "b", "c", "d", "e"],
        }
    ).astype({"A": np.int64, "B": np.float64, "C": object})


def test_optimize_column_int64_to_int8(sample_dataframe):
    col = optimize_column(sample_dataframe["A"])
    assert col.dtype == np.int8


def test_optimize_column_int64_to_int16(sample_dataframe):
    sample_dataframe["A"] = sample_dataframe["A"] + (np.iinfo(np.int8).max + 1)
    col = optimize_column(sample_dataframe["A"])
    assert col.dtype == np.int16


def test_optimize_column_int64_to_int32(sample_dataframe):
    sample_dataframe["A"] = sample_dataframe["A"] + (np.iinfo(np.int16).max + 1)
    col = optimize_column(sample_dataframe["A"])
    assert col.dtype == np.int32


# TODO: Optimize floats with respect to required resolution

# def test_optimize_column_float64_to_float32(sample_dataframe):
#     sample_dataframe["B"] = sample_dataframe["B"] + (np.finfo(np.float16).max + 1.5)
#     col = optimize_column(sample_dataframe["B"])
#     assert col.dtype == np.float32


# def test_optimize_column_float64_to_float16(sample_dataframe):
#     sample_dataframe["B"] = sample_dataframe["B"] + 0.5
#     col = optimize_column(sample_dataframe["B"])
#     assert col.dtype == np.float16


def test_optimize_column_float64_to_int64(sample_dataframe):
    col = sample_dataframe["B"] + (np.iinfo(np.int32).max + 1)
    col = optimize_column(col)
    assert col.dtype == np.int64


def test_optimize_column_float64_to_int32(sample_dataframe):
    col = sample_dataframe["B"] + (np.iinfo(np.int16).max + 1)
    col = optimize_column(col)
    assert col.dtype == np.int32


def test_optimize_column_float64_to_int16(sample_dataframe):
    col = sample_dataframe["B"] + (np.iinfo(np.int8).max + 1)
    col = optimize_column(col)
    assert col.dtype == np.int16


def test_optimize_column_float64_to_int8(sample_dataframe):
    col = optimize_column(sample_dataframe["B"])
    assert col.dtype == np.int8


def test_optimize_column_non_numeric(sample_dataframe):
    col = optimize_column(sample_dataframe["C"])
    updated_dataframe = sample_dataframe.copy()
    updated_dataframe["C"] = col
    sample_dataframe["C"] = sample_dataframe["C"].astype("string[pyarrow]")
    assert updated_dataframe.equals(sample_dataframe)


def test_optimize_column_all_nan(sample_dataframe):
    sample_dataframe["A"] = np.nan
    col = optimize_column(sample_dataframe["A"])
    assert col.dtype == np.float64


def test_optimize_column_geodataframe(sample_dataframe):
    gdf = gpd.GeoDataFrame(
        sample_dataframe,
        geometry=gpd.points_from_xy(sample_dataframe["A"], sample_dataframe["B"]),
    )
    col = optimize_column(gdf["A"])  # pyright: ignore[reportArgumentType]
    assert col.dtype == np.int8
    assert isinstance(col, pd.Series)


def test_optimize_columns_dataframe(sample_dataframe):
    optimized_df = optimize_columns(sample_dataframe)
    for column in optimized_df.columns:
        col = optimized_df[column]
        if isinstance(col, pd.Series):
            assert col.dtype != np.int64
            assert col.dtype != np.float64


def test_optimize_columns_geodataframe(sample_dataframe):
    gdf = gpd.GeoDataFrame(
        sample_dataframe,
        geometry=gpd.points_from_xy(sample_dataframe["A"], sample_dataframe["B"]),
    )
    optimized_gdf = optimize_columns(gdf)
    for column in optimized_gdf.columns:
        col = optimized_gdf[column]
        if isinstance(col, pd.Series):
            assert col.dtype != np.int64
            assert col.dtype != np.float64
    assert isinstance(optimized_gdf, gpd.GeoDataFrame)

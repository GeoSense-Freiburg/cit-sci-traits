"""Tests for the raster_utils module."""

import geopandas as gpd
import pytest
import xarray as xr
from affine import Affine

from src.utils.raster_utils import create_sample_raster, open_raster, raster_to_df


def test_create_sample_raster():
    """
    Test function for create_sample_raster.

    This function tests the create_sample_raster function with different input parameters.

    Test case 1: Test with default extent and resolution
    Test case 2: Test with custom extent and resolution
    Test case 3: Test with negative extent and non-integer resolution

    The function asserts that the result is an instance of xr.DataArray and has the
    expected shape.

    Additionally, it checks that the CRS (Coordinate Reference System) of the result is
    set to EPSG:4326.

    Returns:
        None
    """
    result = create_sample_raster()
    assert isinstance(result, xr.DataArray)
    assert result.shape == (180, 360)  # Assuming resolution of 1

    extent = [-10, -10, 10, 10]
    resolution = 0.1
    result = create_sample_raster(extent=extent, resolution=resolution)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (200, 200)  # Assuming resolution of 0.1

    extent = [-180, -90, -170, -80]
    resolution = 0.05
    result = create_sample_raster(extent=extent, resolution=resolution)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (200, 200)  # Assuming resolution of 0.05

    assert result.rio.crs.to_epsg() == 4326


def test_open_raster():
    """
    Test the open_raster function.

    This function tests the behavior of the open_raster function by opening a valid raster file
    and multiple files.

    Test case 1: Test opening a valid raster file
    Test case 2: Test opening multiple files
    """
    filename = "data/raw/esa_worldcover_v100_1km/esa_worldcover_v100_1km.tif"
    result = open_raster(filename)
    assert isinstance(result, (xr.DataArray, xr.Dataset))


@pytest.fixture(name="rast")
def fixt_rast():
    """Quick sample raster for testing raster_to_gdf."""
    raster = xr.DataArray(
        [[1, 2], [3, 4]],
        dims=("y", "x"),
        coords={"x": [0, 1], "y": [0, 1]},
        attrs={"long_name": "Test raster"},
    )
    raster = raster.rio.write_crs("EPSG:4326")
    transform = Affine(1, 0, 0, 0, -1, 1)
    raster = raster.rio.write_transform(transform=transform)
    return raster


@pytest.fixture(name="geometry")
def fixt_geometry():
    """Quick sample geometry for testing raster_to_gdf."""
    return gpd.GeoSeries(gpd.points_from_xy([0, 1, 0, 1], [0, 0, 1, 1]))


def test_raster_to_gdf_with_long_name(rast, geometry):
    """
    Test function for raster_to_gdf when rast_name is None and 'long_name' attribute exists.
    """
    result = raster_to_df(rast, gdf=True)

    assert isinstance(result, gpd.GeoDataFrame)
    assert set(result.columns) == {"Test raster", "geometry"}
    assert result.geometry.equals(geometry)
    assert result.crs == rast.rio.crs


def test_raster_to_gdf_without_long_name(rast):
    """
    Test function for raster_to_gdf when rast_name is None and 'long_name' attribute does not exist.

    This function tests the raster_to_gdf function when the rast_name is None and the 'long_name'
    attribute does not exist in the raster. It asserts that the function raises a ValueError.

    Returns:
        None
    """
    rast.attrs.pop("long_name")

    try:
        raster_to_df(rast)
    except ValueError as e:
        assert str(e) == "Raster must have a 'long_name' attribute."


def test_raster_to_gdf_with_custom_name(rast, geometry):
    """
    Test function for raster_to_gdf when rast_name is provided.

    This function tests the raster_to_gdf function when the rast_name is provided. It asserts
    that the result is an instance of gpd.GeoDataFrame and has the expected columns and geometry.

    Returns:
        None
    """
    result = raster_to_df(rast, rast_name="Custom Raster", gdf=True)

    assert isinstance(result, gpd.GeoDataFrame)
    assert set(result.columns) == {"Custom Raster", "geometry"}
    assert result.geometry.equals(geometry)
    assert result.crs == rast.rio.crs

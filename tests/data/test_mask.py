"""Tests for the mask module."""

import numpy as np
import xarray as xr

from src.data.mask import get_mask, mask_raster
from src.utils.raster_utils import create_sample_raster


def test_get_mask():
    """Test the get_mask function."""
    ref_raster = create_sample_raster()

    # Define test inputs
    mask_path = "data/raw/esa_worldcover_v100_1km/esa_worldcover_v100_1km.tif"
    keep_classes = [10, 20, 30]

    # Test binary mask
    binary_result = get_mask(mask_path, keep_classes, ref_raster, binary=True)
    assert sorted(
        np.unique(binary_result.values[~np.isnan(binary_result.values)])
    ) == sorted([1])

    # Test non-binary mask
    non_binary_result = get_mask(mask_path, keep_classes, ref_raster, binary=False)
    assert sorted(
        np.unique(non_binary_result.values[~np.isnan(non_binary_result.values)])
    ) == sorted(keep_classes)


def test_mask_raster():
    """Test the mask_raster function."""
    # Create sample data
    rast_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mask_data = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    rast = xr.DataArray(rast_data)
    mask = xr.DataArray(mask_data)

    # Apply mask
    masked_rast = mask_raster(rast, mask)

    # Check masked values
    expected_result = np.array([[1, np.nan, 3], [np.nan, 5, np.nan], [7, np.nan, 9]])
    np.testing.assert_array_equal(
        masked_rast.values, expected_result  # pyright: ignore[reportArgumentType]
    )

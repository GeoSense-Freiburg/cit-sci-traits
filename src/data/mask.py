import os

import rioxarray as riox
import xarray as xr

from src.utils.raster_utils import open_raster


def mask_non_vegetation(
    rast: xr.DataArray | xr.Dataset, mask: str | os.PathLike
) -> xr.DataArray | xr.Dataset:
    mask_rast = open_raster(mask, mask_and_scale=True)
    # Mask is ESA WorldCover v100
    # Mask rast by mask_rast wherever mask_rast is not vegetation
    mask_rast = riox.reproject_match()

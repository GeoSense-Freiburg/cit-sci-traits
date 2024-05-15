import xarray as xr

from src.utils.raster_utils import create_sample_raster, open_raster


def mask_raster(rast: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """
    Mask out non-vegetation classes from a raster.

    Parameters:
        rast (xr.DataArray or xr.Dataset): The input raster data to be masked.
        mask (xr.DataArray): The mask indicating vegetation classes (1 for vegetation,
            0 for non-vegetation).

    Returns:
        xr.DataArray or xr.Dataset: The masked raster data, where non-vegetation classes
            are set to NaN.

    """
    rast = rast.where(mask > 0)
    return rast


def get_mask(
    mask_path: str,
    keep_classes: list[int],
    resolution: int | float,
    binary: bool = True,
) -> xr.DataArray:
    """
    Generate a mask based on the given mask path, list of classes to keep, and reference
        raster.

    Args:
        mask_path (str): The path to the mask file.
        keep_classes (list[int]): A list of classes to keep in the mask.
        ref_raster (xr.Dataset): The reference raster used for reprojection.
        binary (bool, optional): Whether to convert the mask to a binary mask. Defaults
            to True.

    Returns:
        xr.DataArray: The generated mask.

    """
    ref_raster = create_sample_raster(resolution=resolution)

    mask = (
        open_raster(mask_path)
        .sel(band=1)
        .where(lambda x: x.isin(keep_classes))
        .rio.reproject_match(ref_raster)
    )

    if binary:
        return (
            mask.where(mask.notnull(), 0)
            .where(lambda x: x == 0, 1)
            .where(lambda x: x == 1)
        )

    return mask

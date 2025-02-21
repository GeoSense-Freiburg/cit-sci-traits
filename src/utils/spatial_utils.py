"""Spatial utility functions."""

from typing import Iterable

import cupy as cp
import h3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pyproj import Proj
from scipy.spatial import KDTree
from shapely.geometry import shape


def get_h3_resolution(edge_length: float) -> int | float:
    """
    Calculates the H3 resolution based on the given edge length.

    Edge lengths according to the H3 documentation:
    https://h3geo.org/docs/core-library/restable/#edge-lengths

    Parameters:
        edge_length (float): The length of the H3 hexagon edge.

    Returns:
        int | float: The H3 resolution corresponding to the given edge length.
    """
    edge_lengths = np.array(
        [
            1281.256011,
            483.0568391,
            182.5129565,
            68.97922179,
            26.07175968,
            9.854090990,
            3.724532667,
            1.406475763,
            0.531414010,
            0.200786148,
            0.075863783,
            0.028663897,
            0.010830188,
            0.004092010,
            0.001546100,
            0.000584169,
        ]
    )

    resolutions = np.arange(len(edge_lengths))

    # Fit a logarithmic function to the data
    coeffs = np.polyfit(np.log(edge_lengths), resolutions, deg=1)

    return np.polyval(coeffs, np.log(edge_length))


def get_edge_length(r: int | float) -> int | float:
    """
    Calculate the edge length of an equilateral triangle given the apothem length.

    Parameters:
    r (int | float): The apothem length of the equilateral triangle in meters

    Returns:
    int | float: The edge length of the equilateral triangle in kilometers
    """
    return (r * 2 / 1000) / np.sqrt(3)


def acr_to_h3_res(acr: int | float) -> int | float:
    """
    Converts an autocorrelation range (ACR) to the corresponding H3 resolution.

    Parameters:
    acr (int | float): The autocorrelation range.

    Returns:
    int | float: The H3 resolution corresponding to the given ACR.
    """
    return get_h3_resolution(get_edge_length(acr / 2))


def assign_hexagons(
    df: pd.DataFrame,
    resolution: int | float,
    lat: str = "y",
    lon: str = "x",
    dask: bool = False,
) -> pd.DataFrame:
    """
    Assigns hexagon IDs to a DataFrame based on latitude and longitude coordinates.

    Args:
        df (pd.DataFrame): The DataFrame containing latitude and longitude coordinates.
        resolution (int | float): The resolution of the hexagons.
        lat (str, optional): The name of the latitude column in the DataFrame.
            Defaults to "y".
        lon (str, optional): The name of the longitude column in the DataFrame.
            Defaults to "x".
        dask (bool, optional): Whether to use Dask for parallel processing.
            Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame with an additional column "hex_id" containing the
            assigned hexagon IDs.
    """

    def _assign_hex_to_df(_df: pd.DataFrame) -> pd.DataFrame:
        if _df.empty:
            _df["hex_id"] = np.nan
            return _df
        _df = _df.copy()
        geo_to_h3_vectorized = np.vectorize(h3.geo_to_h3, otypes=[str])
        _df.loc[:, "hex_id"] = geo_to_h3_vectorized(_df[lat], _df[lon], resolution)
        return _df

    if dask:
        meta = df._meta.assign(hex_id=pd.Series(dtype="string"))
        return df.map_partitions(
            _assign_hex_to_df, meta=meta
        )  # pyright: ignore[reportCallIssue]

    return _assign_hex_to_df(df)


def get_lat_area(lat: int | float, resolution: int | float) -> float:
    """Calculate the area of a grid cell at a given latitude."""
    # Define the grid cell coordinates
    coordinates = [
        (0, lat + (resolution / 2)),
        (resolution, lat + (resolution / 2)),
        (resolution, lat - (resolution / 2)),
        (0, lat - (resolution / 2)),
        (0, lat + (resolution / 2)),  # Close the polygon by repeating the first point
    ]

    # Define the projection string directly using the coordinates
    projection_string = (
        f"+proj=aea +lat_1={coordinates[0][1]} +lat_2={coordinates[2][1]} "
        f"+lat_0={lat} +lon_0={resolution / 2}"
    )
    pa = Proj(projection_string)

    # Project the coordinates and create the polygon
    x, y = pa(*zip(*coordinates))  # pylint: disable=unpacking-non-sequence
    area = shape({"type": "Polygon", "coordinates": [list(zip(x, y))]}).area / 1000000

    return area


def lat_weights(lat_unique: Iterable[int | float], resolution: int | float) -> dict:
    """Calculate weights for each latitude band based on area of grid cells."""
    weights = {}

    for j in lat_unique:
        weights[j] = get_lat_area(j, resolution)

    # Normalize the weights by the maximum area
    max_area = max(weights.values())
    weights = {k: v / max_area for k, v in weights.items()}

    return weights


def weighted_pearson_r(df: pd.DataFrame, weights: dict) -> float:
    """Calculate the weighted Pearson correlation coefficient between two DataFrames."""
    df["weights"] = df.index.get_level_values("y").map(weights)
    model = sm.stats.DescrStatsW(df.iloc[:, :2], df["weights"])
    return model.corrcoef[0, 1]


def interpolate_like(
    decimated: np.ndarray,
    reference_valid_mask: np.ndarray,
    method="nearest",
    use_gpu=False,
    nodata_value=None,
) -> np.ndarray:
    """
    Interpolate a decimated raster to match the extent of a reference, with optional GPU acceleration.

    Parameters:
    decimated (np.ndarray): The decimated raster array where values may include NaN or nodata values.
    reference_valid_mask (np.ndarray): A boolean mask indicating valid reference locations.
    method (str): Interpolation method, either 'nearest' or 'bilinear'. Defaults to 'nearest'.
    use_gpu (bool): Whether to use GPU acceleration with CuPy. Defaults to False.
    nodata_value (float or int, optional): Value to treat as nodata, which will be replaced by NaN. Defaults to None.

    Returns:
    np.ndarray: The interpolated raster array with the same shape as the input, where NaN values
                in the decimated array are filled based on the specified interpolation method.
    """
    # Use CuPy if GPU acceleration is enabled
    xp = cp if use_gpu else np

    if use_gpu:
        decimated = cp.asarray(decimated)
        reference_valid_mask = cp.asarray(reference_valid_mask)

    # Convert nodata values to NaN if a nodata_value is provided
    if nodata_value is not None:
        decimated = xp.where(decimated == nodata_value, xp.nan, decimated)

    # Identify NaN mask and interpolation mask
    nan_mask = xp.isnan(decimated)

    # Interplation mask indicates where we have a valid reference AND a NaN in the
    # decimated raster
    interpolation_mask = nan_mask & reference_valid_mask

    # Create coordinate grids
    x_coords, y_coords = xp.meshgrid(
        xp.arange(decimated.shape[1]), xp.arange(decimated.shape[0])
    )

    # Get valid points
    valid_x = x_coords[~nan_mask].ravel()
    valid_y = y_coords[~nan_mask].ravel()
    valid_values = decimated[~nan_mask].ravel()

    if method == "nearest":
        if use_gpu:
            # GPU-accelerated nearest-neighbor using CuPy's distance computations
            interp_x = x_coords[interpolation_mask].ravel()
            interp_y = y_coords[interpolation_mask].ravel()

            query_points = xp.stack([interp_x, interp_y], axis=-1)
            valid_points = xp.stack([valid_x, valid_y], axis=-1)

            distances = cp.linalg.norm(
                query_points[:, xp.newaxis, :] - valid_points[xp.newaxis, :, :], axis=2
            )
            nearest_indices = distances.argmin(axis=1)
            interpolated_values = valid_values[nearest_indices]
        else:
            # CPU-based nearest-neighbor using KDTree
            tree = KDTree(xp.c_[valid_x, valid_y])
            interp_x = x_coords[interpolation_mask].ravel()
            interp_y = y_coords[interpolation_mask].ravel()
            _, indices = tree.query(xp.c_[interp_x, interp_y], k=1)
            interpolated_values = valid_values[indices]

    elif method == "bilinear":
        raise NotImplementedError("Bilinear interpolation is not yet supported.")
        # The below code is a work-in-progress implementation of bilinear interpolation
        # Right now it is not working as expected, so it is commented out
        # Bilinear interpolation
        # interp_x = x_coords[interpolation_mask]
        # interp_y = y_coords[interpolation_mask]

        # # Find integer grid coordinates surrounding the interpolation points
        # x0 = xp.floor(interp_x).astype(int)
        # x1 = xp.ceil(interp_x).astype(int)
        # y0 = xp.floor(interp_y).astype(int)
        # y1 = xp.ceil(interp_y).astype(int)

        # # Clip to ensure indices are within bounds
        # x0 = xp.clip(x0, 0, decimated.shape[1] - 1)
        # x1 = xp.clip(x1, 0, decimated.shape[1] - 1)
        # y0 = xp.clip(y0, 0, decimated.shape[0] - 1)
        # y1 = xp.clip(y1, 0, decimated.shape[0] - 1)

        # # Extract the values at the four surrounding points
        # q11 = decimated[y0, x0]
        # q21 = decimated[y0, x1]
        # q12 = decimated[y1, x0]
        # q22 = decimated[y1, x1]

        # # Replace NaNs with 0 for interpolation (or use masking logic if NaNs shouldn't contribute)
        # q11 = xp.where(xp.isnan(q11), 0, q11)
        # q21 = xp.where(xp.isnan(q21), 0, q21)
        # q12 = xp.where(xp.isnan(q12), 0, q12)
        # q22 = xp.where(xp.isnan(q22), 0, q22)

        # # Compute the weights for the bilinear interpolation
        # wx = interp_x - x0
        # wy = interp_y - y0

        # # Compute the interpolated value
        # interpolated_values = (
        #     (1 - wx) * (1 - wy) * q11
        #     + wx * (1 - wy) * q21
        #     + (1 - wx) * wy * q12
        #     + wx * wy * q22
        # )

    else:
        raise ValueError("Unsupported interpolation method. Use 'nearest'.")

    # Fill the raster
    filled_raster = decimated.copy()
    filled_raster[interpolation_mask] = interpolated_values

    # Convert back to NumPy if using CuPy
    if use_gpu:
        filled_raster = cp.asnumpy(filled_raster)

    return filled_raster

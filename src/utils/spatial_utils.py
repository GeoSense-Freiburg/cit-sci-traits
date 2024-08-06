"""Spatial utility functions."""

from typing import Iterable
import h3
import numpy as np
import pandas as pd
from pyproj import Proj
from shapely.geometry import shape
import statsmodels.api as sm


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
        geo_to_h3_vectorized = np.vectorize(h3.geo_to_h3)
        _df["hex_id"] = geo_to_h3_vectorized(_df[lat], _df[lon], resolution)
        return _df

    if dask:
        return df.map_partitions(_assign_hex_to_df)

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

import pickle

import numpy as np
import pandas as pd

from src.utils.dataset_utils import get_power_transformer_fn
from src.utils.trait_utils import get_trait_number_from_id


def yeo_johnson_transform(x: np.ndarray, lmbda: int | float) -> np.ndarray:
    """Return transformed input x following Yeo-Johnson transform with
    parameter lambda.

    Source: from sklearn.preprocessing._data.py
    """

    out = np.zeros_like(x)
    pos = x >= 0  # binary mask

    # when x >= 0
    if abs(lmbda) < np.spacing(1.0):
        out[pos] = np.log1p(x[pos])
    else:  # lmbda != 0
        out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.0):
        out[~pos] = -(np.power(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
    else:  # lmbda == 2
        out[~pos] = -np.log1p(-x[~pos])

    return out


def yeo_johnson_inverse_transform(x: np.ndarray, lmbda: int | float) -> np.ndarray:
    """Return inverse-transformed input x following Yeo-Johnson inverse
    transform with parameter lambda.

    Source: from sklearn.preprocessing._data.py
    """
    x_inv = np.zeros_like(x)
    pos = x >= 0
    # Small epsilon for numerical stability (avoid raising a negative number to a fractional power)
    eps = np.finfo(float).eps

    # when x >= 0
    if abs(lmbda) < np.spacing(1.0):
        x_inv[pos] = np.exp(x[pos]) - 1
    else:  # lmbda != 0
        base = x[pos] * lmbda + 1
        x_inv[pos] = np.power(np.maximum(base, eps), 1 / lmbda) - 1

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.0):
        base = -(2 - lmbda) * x[~pos] + 1
        x_inv[~pos] = 1 - np.power(np.maximum(base, eps), 1 / (2 - lmbda))
    else:  # lmbda == 2
        x_inv[~pos] = 1 - np.exp(-x[~pos])

    return x_inv


def power_back_transform(data: np.ndarray, trait_num: str) -> np.ndarray:
    """Back-transform power-transformed data."""
    with open(get_power_transformer_fn(), "rb") as xf:
        xfer = pickle.load(xf)

        nan_mask = None
        if np.isnan(data).any():
            nan_mask = np.isnan(data)

        shape = data.shape
        if len(shape) > 1:
            data = data.ravel()

        feature_nums = np.array(
            [get_trait_number_from_id(f) for f in xfer.feature_names_in_]
        )
        ft_id = np.where(feature_nums == trait_num)[0][0]
        data_bt = xfer.inverse_transform(
            pd.DataFrame(columns=xfer.feature_names_in_)
            .assign(**{f"X{trait_num}": data})
            .fillna(0)
        )[:, ft_id]

        if len(shape) > 1:
            data_bt = data_bt.reshape(data.shape)

        if nan_mask is not None:
            data_bt[nan_mask] = np.nan

        return data_bt


def power_transform(data: np.ndarray, trait_num: str) -> np.ndarray:
    """Apply a power transform to new trait data using a pickled transformer.

    The transformation uses the same lambda values as the existing transformer.

    Parameters:
        data (np.ndarray): The new trait data to transform.
        trait_num (str): The trait number (as a string) to indicate which column to transform.

    Returns:
        np.ndarray: The power-transformed data.
    """
    with open(get_power_transformer_fn(), "rb") as xf:
        xfer = pickle.load(xf)

        nan_mask = None
        if np.isnan(data).any():
            nan_mask = np.isnan(data)

        shape = data.shape
        if len(shape) > 1:
            data = data.ravel()

        feature_nums = np.array(
            [get_trait_number_from_id(f) for f in xfer.feature_names_in_]
        )
        ft_id = np.where(feature_nums == trait_num)[0][0]

        # Create an empty DataFrame with the transformer's feature names as columns,
        # assign the new trait data to the appropriate column, and fill missing values with 0.
        df = (
            pd.DataFrame(columns=xfer.feature_names_in_)
            .assign(**{f"X{trait_num}": data})
            .fillna(0)
        )
        transformed = xfer.transform(df)[:, ft_id]

        if len(shape) > 1:
            transformed = transformed.reshape(shape)

        if nan_mask is not None:
            transformed[nan_mask] = np.nan

        return transformed

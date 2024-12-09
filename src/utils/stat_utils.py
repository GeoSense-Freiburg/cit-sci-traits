import numpy as np


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

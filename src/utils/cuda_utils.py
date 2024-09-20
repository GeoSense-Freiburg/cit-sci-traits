import cudf
import cupy as cp
import pandas as pd


def df_to_cupy(df: pd.DataFrame, device_id: int) -> cp.ndarray:
    """Convert a Pandas DataFrame to CuPy array on a specific GPU device."""
    with cp.cuda.Device(device_id):
        return cudf.DataFrame.from_pandas(df).to_cupy()


def df_to_cudf(df: pd.DataFrame, device_id: int) -> cudf.DataFrame:
    """Convert a Pandas DataFrame to CuDF DataFrame on a specific GPU device."""
    with cp.cuda.Device(device_id):
        return cudf.DataFrame.from_pandas(df)

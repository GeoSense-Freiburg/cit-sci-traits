"""Utility functions for Dask. For Dask-CUDA, see src/utils/dask_cuda_utils.py."""

from typing import Any

import dask.dataframe as dd
import pandas as pd
from distributed import Client, LocalCluster


def init_dask(**kwargs) -> tuple[Client, LocalCluster]:
    """Initialize the Dask client and cluster."""
    cluster = LocalCluster(**kwargs)

    client = Client(cluster)
    return client, cluster


def close_dask(client: Client, cluster: LocalCluster | Any) -> None:
    """Close the Dask client and cluster."""
    client.close()
    cluster.close()


def df_to_dd(
    df: pd.DataFrame, npartitions: int
) -> dd.DataFrame:  # pyright: ignore[reportPrivateImportUsage]
    """Convert a Pandas DataFrame to a Dask DataFrame."""
    return dd.from_pandas(  # pyright: ignore[reportPrivateImportUsage]
        df, npartitions=npartitions
    )

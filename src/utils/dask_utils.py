"""Utility functions for Dask. For Dask-CUDA, see src/utils/dask_cuda_utils.py."""

import os
import signal
from typing import Any

import dask.dataframe as dd
import pandas as pd
from dask.distributed import TimeoutError
from distributed import Client, LocalCluster


def init_dask(**kwargs) -> tuple[Client, LocalCluster]:
    """Initialize the Dask client and cluster."""
    cluster = LocalCluster(**kwargs)

    client = Client(cluster)
    return client, cluster


def close_dask(client: Client, cluster: LocalCluster | Any, timeout: int = 5) -> None:
    """Close the Dask client and cluster with a timeout."""
    try:
        client.close()
        cluster.close()
    except TimeoutError:
        print(
            f"Warning: Cluster did not close within {timeout} seconds. "
            "Forcefully terminating."
        )
        os.kill(int(cluster.scheduler_address), signal.SIGKILL)
        return


def df_to_dd(
    df: pd.DataFrame, npartitions: int
) -> dd.DataFrame:  # pyright: ignore[reportPrivateImportUsage]
    """Convert a Pandas DataFrame to a Dask DataFrame."""
    return dd.from_pandas(  # pyright: ignore[reportPrivateImportUsage]
        df, npartitions=npartitions
    )

import dask.dataframe as dd
import pandas as pd
from dask_cuda import LocalCUDACluster
from distributed import Client, LocalCluster


def init_dask(
    cuda: bool = False,
    device_ids: tuple[int, ...] | None = None,
    memory_limit: str = "auto",
) -> tuple[Client, LocalCluster]:
    """Initialize the Dask client and cluster."""
    if cuda:
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=device_ids, memory_limit=memory_limit
        )
    else:
        cluster = LocalCluster(memory_limit=memory_limit)

    client = Client(cluster)
    return client, cluster


def close_dask(client: Client, cluster: LocalCluster | LocalCUDACluster) -> None:
    """Close the Dask client and cluster."""
    client.close()
    cluster.close()


def df_to_dd(
    df: pd.DataFrame, npartitions: int
) -> dd.DataFrame:  # pyright: ignore[reportPrivateImportUsage]
    return dd.from_pandas(  # pyright: ignore[reportPrivateImportUsage]
        df, npartitions=npartitions
    )

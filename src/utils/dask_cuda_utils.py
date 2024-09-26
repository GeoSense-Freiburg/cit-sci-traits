"""Utility functions for Dask-CUDA."""

from dask.distributed import Client
from dask_cuda import LocalCUDACluster


def init_dask_cuda(
    device_ids: tuple[int, ...] | None = None,
    memory_limit: str = "auto",
) -> tuple[Client, LocalCUDACluster]:
    """Initialize the Dask CUDA client and cluster."""

    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES=device_ids, memory_limit=memory_limit
    )

    client = Client(cluster)
    return client, cluster

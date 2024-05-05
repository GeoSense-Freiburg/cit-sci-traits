"""Subsample GBIF data and save to disk."""

from pathlib import Path
from typing import Callable

import dask.dataframe as dd
import h3
import numpy as np
import pandas as pd
from box import ConfigBox
from dask import config
from dask.diagnostics import ProgressBar
from dask.distributed import Client

from src.conf.conf import get_config


def main(cfg: ConfigBox = get_config()) -> None:
    """Subsample GBIF data and save to disk."""

    # 00. Define helper functions
    def _lat_lon_to_hex(lat: float, lon: float, resolution: int = 4):
        return int(h3.geo_to_h3(lat, lon, resolution), 16)

    def _apply_hex_to_partition(df, lat_lon_to_hex_vectorized: Callable):
        return df.assign(
            hex=lat_lon_to_hex_vectorized(
                df.decimallatitude,
                df.decimallongitude,
                cfg.interim.gbif.subsample_binsize,
            )
        )

    def _sample_partition(df, n_samples: int = cfg.interim.gbif.subsample_n_max):
        return df.groupby(level=0).apply(
            lambda group: (
                group
                if len(group) <= n_samples
                else group.sample(n_samples, random_state=cfg.random_seed)
            )
        )

    # 01. Initialize Dask client
    with config.set(
        {
            "distributed.worker.memory.target": 0.70,
            "distributed.worker.memory.spill": 0.80,
            "distributed.worker.memory.pause": 0.90,
            "distributed.worker.memory.terminate": 0.98,
        }
    ):
        client = Client(n_workers=24, memory_limit="120GB")

    # 02. Load GBIF data
    gbif_prep_dir = Path(cfg.interim.gbif.dir)
    gbif = (
        dd.read_parquet(gbif_prep_dir / cfg.interim.gbif.matched)
        .repartition(npartitions=64)
        .astype({"pft": "category", "speciesname": "string[pyarrow]"})
    )

    # 03. Vectorize lat_lon_to_hex for better performance
    lat_lon_to_hex_vectorized = np.vectorize(_lat_lon_to_hex)

    # 04. Apply hex binning to GBIF data using Dask
    meta = gbif._meta.assign(  # pylint: disable=protected-access
        hex=pd.Series(dtype="string")
    )

    gbif_hex_binned = (
        gbif.sort_values(["decimallatitude", "decimallongitude"]).map_partitions(
            _apply_hex_to_partition, lat_lon_to_hex_vectorized, meta=meta
        )
        # .sort_values("hex")
        .set_index("hex", npartitions=64, shuffle="disk")
    )

    meta = gbif_hex_binned._meta_nonempty.sample(0)  # pylint: disable=protected-access

    # 05. Compute and save subsampled GBIF d[ata
    try:
        with ProgressBar():
            gbif_hex_binned.map_partitions(_sample_partition, meta=meta).reset_index(
                drop=True
            ).to_parquet(gbif_prep_dir / cfg.interim.gbif.subsampled, write_index=False)
    finally:
        client.close()


if __name__ == "__main__":
    main()

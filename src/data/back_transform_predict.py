import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from rioxarray import open_rasterio

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import get_power_transformer_fn, get_predict_dir
from src.utils.raster_utils import pack_xr, xr_to_raster_rasterio
from src.utils.trait_utils import get_trait_number_from_id

if __name__ == "__main__":
    cfg = get_config()

    client, cluster = init_dask(
        n_workers=32, threads_per_worker=2, dashboard_address=cfg.dask_dashboard
    )

    fns = [
        list(Path(d, "splot_gbif").glob("*.tif"))[0]
        for d in get_predict_dir().iterdir()
    ]

    def process_chunk(chunk: np.ndarray, trait_num: str) -> np.ndarray:
        with open(get_power_transformer_fn(cfg), "rb") as xf:
            xfer = pickle.load(xf)

            nan_mask = np.isnan(chunk)

            feature_nums = np.array(
                [get_trait_number_from_id(f) for f in xfer.feature_names_in_]
            )
            ft_id = np.where(feature_nums == trait_num)[0][0]
            chunk_bt = xfer.inverse_transform(
                pd.DataFrame(columns=xfer.feature_names_in_)
                .assign(**{f"X{trait_num}": chunk.ravel()})
                .fillna(0)
            )[:, ft_id]
            chunk_bt = chunk_bt.reshape(chunk.shape)
            chunk_bt[nan_mask] = np.nan
            return chunk_bt

    for fn in fns:
        log.info(f"Processing {fn}")
        r = open_rasterio(
            fn, chunks={"band": 1, "x": 1000, "y": 1000}, mask_and_scale=True
        ).sel(band=1)
        trait_num = get_trait_number_from_id(fn.stem)
        bt = xr.apply_ufunc(
            process_chunk, r, trait_num, dask="parallelized", output_dtypes=[float]
        ).compute()

        log.info(f"Writing back-transformed raster to {fn}")
        xr_to_raster_rasterio(pack_xr(bt), fn)
        r.close()

    close_dask(client, cluster)

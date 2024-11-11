from pathlib import Path

import pandas as pd
from dask import compute, delayed

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import get_aoa_dir
from src.utils.raster_utils import open_raster


def main() -> None:
    cfg = get_config()

    log.info("Gathering filenames...")
    splot_fns = [
        list(Path(d, "splot").glob("*.tif"))[0]
        for d in sorted(list(get_aoa_dir(cfg).glob("*")))
        if d.is_dir()
    ]

    comb_fns = [
        list(Path(d, "splot_gbif").glob("*.tif"))[0]
        for d in sorted(list(get_aoa_dir(cfg).glob("*")))
        if d.is_dir()
    ]

    @delayed
    def _aoa_frac(fn: Path) -> tuple[str, float]:
        ds = open_raster(fn).sel(band=2)
        frac = 1 - (ds == 1).sum().values / (ds == 0).sum().values
        ds.close()
        del ds
        return fn.parents[1].stem, frac

    # Initalize dask
    client, cluster = init_dask(dashboard_address=cfg.dask_dashboard)

    log.info("Computing sPlot AOA fractions...")
    splot_aoa_fracs = compute(*[_aoa_frac(fn) for fn in splot_fns])
    log.info("Computing combined AOA fractions...")
    comb_aoa_fracs = compute(*[_aoa_frac(fn) for fn in comb_fns])

    # Close dask
    close_dask(client, cluster)

    log.info("Updating results...")
    all_results = pd.read_parquet("results/all_results.parquet")

    for trait_id, aoa in splot_aoa_fracs:
        rows = all_results.query(
            f"trait_id == '{trait_id}' and "
            "trait_set == 'splot' and "
            f"resolution == '{cfg.model_res}'"
        )
        all_results.loc[rows.index, "aoa"] = aoa

    for trait_id, aoa in comb_aoa_fracs:
        rows = all_results.query(
            f"trait_id == '{trait_id}' and "
            "trait_set == 'splot_gbif' and "
            f"resolution == '{cfg.model_res}'"
        )
        all_results.loc[rows.index, "aoa"] = aoa

    # Back up the results
    log.info("Backing up results...")
    Path("results/all_results.parquet").rename("results/all_results.parquet.bak")

    log.info("Saving updated results...")
    all_results.to_parquet("results/all_results.parquet")

    log.info("Done!")


if __name__ == "__main__":
    main()

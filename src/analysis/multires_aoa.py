import shutil
from pathlib import Path

import pandas as pd
from box import BoxKeyError
from dask import compute, delayed

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import get_aoa_dir
from src.utils.raster_utils import open_raster


def main() -> None:
    cfg = get_config()
    transform = getattr(cfg.trydb.interim, "transform", "none")

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
    all_aoa_fn = Path("results/all_aoa.parquet")
    if all_aoa_fn.exists():
        all_aoa = pd.read_parquet(all_aoa_fn)
        # Back up the results
        log.info("Backing up results...")
        shutil.copy("results/all_aoa.parquet", "results/all_aoa.parquet.bak")
    else:
        all_aoa = pd.DataFrame(
            columns=[
                "pft",
                "resolution",
                "trait_id",
                "trait_set",
                "transform",
                "aoa",
            ]
        )

    df_splot = pd.DataFrame(
        {
            "pft": cfg.PFT,
            "resolution": cfg.model_res,
            "trait_id": [s[0] for s in splot_aoa_fracs],
            "trait_set": "splot",
            "transform": transform,
            "aoa": [s[1] for s in splot_aoa_fracs],
        }
    )

    df_comb = pd.DataFrame(
        {
            "pft": cfg.PFT,
            "resolution": cfg.model_res,
            "trait_id": [s[0] for s in comb_aoa_fracs],
            "trait_set": "splot_gbif",
            "transform": transform,
            "aoa": [s[1] for s in comb_aoa_fracs],
        }
    )

    all_aoa = pd.concat(
        [all_aoa, df_splot, df_comb], ignore_index=True
    ).drop_duplicates(
        subset=["pft", "resolution", "trait_id", "trait_set", "transform"],
        ignore_index=True,
    )

    log.info("Saving updated results...")
    all_aoa.to_parquet("results/all_aoa.parquet")

    log.info("Done!")


if __name__ == "__main__":
    main()

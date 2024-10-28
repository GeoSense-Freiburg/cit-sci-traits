"""Match GBIF and PFT data and save to disk."""

from pathlib import Path

import dask.dataframe as dd
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.dask_utils import close_dask, init_dask
from src.utils.trait_utils import clean_species_name


def main(cfg: ConfigBox = get_config()):
    """Match GBIF and PFT data and save to disk."""
    syscfg = cfg[detect_system()]["match_gbif_pfts"]
    # 00. Initialize Dask client
    client, cluster = init_dask(
        dashboard_address=cfg.dask_dashboard, n_workers=syscfg.n_workers
    )

    # 01. Load data
    gbif_raw_dir = Path(cfg.raw_dir, cfg.gbif.raw.dir)
    gbif_prep_dir = Path(cfg.interim_dir, cfg.gbif.interim.dir)

    columns = [
        "species",
        "taxonrank",
        "decimallatitude",
        "decimallongitude",
    ]
    ddf = dd.read_parquet(
        gbif_raw_dir / "all_tracheophyta_non-cult_2024-04-10.parquet/*", columns=columns
    )

    pfts = dd.read_csv(Path(cfg.raw_dir, cfg.trydb.raw.pfts), encoding="latin-1")

    # 02. Preprocess GBIF data
    ddf = (
        ddf[ddf["taxonrank"] == "SPECIES"]
        .drop(columns=["taxonrank"])
        .dropna(subset=["species"])
        .pipe(clean_species_name, "species", "speciesname")
        .drop(columns=["species"])
        .set_index("speciesname")
    )

    # 03. Preprocess PFT data
    pfts = (
        pfts.drop(columns=["AccSpeciesID"])
        .dropna(subset=["AccSpeciesName"])
        .pipe(clean_species_name, "AccSpeciesName", "speciesname")
        .drop(columns=["AccSpeciesName"])
        .set_index("speciesname")
    )

    log.info("Matching GBIF and PFT data and saving to disk...")
    # 04. Merge GBIF and PFT data and save to disk
    try:
        ddf = (
            ddf.join(pfts, how="inner")
            .reset_index()
            .to_parquet(gbif_prep_dir / cfg.gbif.interim.matched, write_index=False)
        )
    finally:
        log.info("Shutting down Dask client...")
        close_dask(client, cluster)


if __name__ == "__main__":
    main()
    log.info("Done!")

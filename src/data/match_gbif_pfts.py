"""Match GBIF and PFT data and save to disk."""

from pathlib import Path

import dask.dataframe as dd
from box import ConfigBox
from dask.distributed import Client, LocalCluster

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.trait_utils import clean_species_name


def main(cfg: ConfigBox = get_config()):
    """Match GBIF and PFT data and save to disk."""
    # 00. Initialize Dask client
    cluster = LocalCluster(
        dashboard_address=":39143", n_workers=40, memory_limit="48GB"
    )
    client = Client(cluster)

    # 01. Load data
    gbif_raw_dir = Path(cfg.gbif.raw.dir)
    gbif_prep_dir = Path(cfg.interim.gbif.dir)

    columns = [
        "species",
        "taxonrank",
        "decimallatitude",
        "decimallongitude",
    ]
    ddf = dd.read_parquet(
        gbif_raw_dir / "all_tracheophyta_non-cult_2024-04-10.parquet/*",
        columns=columns,
    ).repartition(npartitions=60)

    pfts = dd.read_csv(Path(cfg.trydb.raw.pfts), encoding="latin-1")

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
            .to_parquet(gbif_prep_dir / cfg.interim.gbif.matched, write_index=False)
        )
    finally:
        log.info("Shutting down Dask client...")
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
    log.info("Done!")

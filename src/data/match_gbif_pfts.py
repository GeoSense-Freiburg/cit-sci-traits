"""Match GBIF and PFT data and save to disk."""

from pathlib import Path

import dask.dataframe as dd
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import log


def main(cfg: ConfigBox = get_config()):
    """Match GBIF and PFT data and save to disk."""
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
        npartitions=60,
    )
    pfts = dd.read_csv(Path(cfg.trydb.raw.pfts), encoding="latin-1")

    # 02. Preprocess GBIF data
    ddf = (
        ddf[ddf["taxonrank"] == "SPECIES"]
        .drop(columns=["taxonrank"])
        .assign(
            speciesname=lambda ddf: ddf.species.str.extract(
                "([A-Za-z]+ [A-Za-z]+)", expand=False
            )
        )
        .dropna(subset=["speciesname"])
        .drop(columns=["species"])
        .set_index("speciesname")
    )

    # 03. Preprocess PFT data
    pfts = pfts.drop(columns=["AccSpeciesID"]).assign(
        speciesname=lambda df: df.AccSpeciesName.str.extract(
            "([A-Za-z]+ [A-Za-z]+)", expand=False
        )
        .dropna(subset="speciesname")
        .drop(columns=["AccSpeciesName"])
        .set_index("speciesname")
    )

    log.info("Matching GBIF and PFT data and saving to disk...")
    # 04. Merge GBIF and PFT data and save to disk
    ddf = (
        dd.merge(ddf, pfts, left_index=True, right_index=True)
        .reset_index()
        .to_parquet(gbif_prep_dir / cfg.interim.gbif.matched, write_index=False)
    )


if __name__ == "__main__":
    main()

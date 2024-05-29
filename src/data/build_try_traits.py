"""Extracts, cleans, and gets species mean trait values from TRY data."""

import shutil
import zipfile
from pathlib import Path

import pandas as pd
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.df_utils import filter_outliers
from src.utils.trait_utils import clean_species_name


def main(cfg: ConfigBox = get_config()) -> None:
    """Extract, clean, and get species mean trait values from TRY data."""
    try_raw_dir = Path(cfg.raw_dir, cfg.trydb.raw.dir)
    try_prep_dir = Path(cfg.interim_dir, cfg.trydb.interim.dir)
    ext_traits_fp = try_prep_dir / cfg.trydb.raw.zipfile_csv
    traits_fp = try_prep_dir / cfg.trydb.interim.extracted_csv

    log.info("Extracting raw TRY traits data...")
    with zipfile.ZipFile(try_raw_dir / cfg.trydb.raw.zip, "r") as zip_ref:
        try_prep_dir.mkdir(parents=True, exist_ok=True)
        zip_ref.extract(
            cfg.trydb.raw.zipfile_csv,
            try_prep_dir,
        )
    ext_traits_fp.rename(traits_fp)

    log.info("Filtering outliers and getting species mean trait values...")
    traits = pd.read_csv(traits_fp, encoding="latin-1", index_col=0)

    drop_cols = ["Unnamed: 5", "Genus", "Family", "ObservationID"]
    mean_filt_traits = (
        traits.drop(columns=drop_cols)
        .pipe(
            lambda df: filter_outliers(
                df,
                cols=df.filter(like="X").columns.to_list(),
                quantiles=tuple(cfg.trydb.interim.quantile_range),
            )
        )
        .pipe(clean_species_name, "Species", "speciesname")
        .drop(columns=["Species"])
        .groupby("speciesname")
        .mean()
        .reset_index()
    )

    log.info("Saving filtered and mean trait values...")
    mean_filt_traits.to_parquet(
        try_prep_dir / cfg.trydb.interim.filtered, index=False, compression="zstd"
    )

    log.info("Cleaning up...")
    shutil.rmtree(try_prep_dir / Path(cfg.trydb.raw.zipfile_csv).parents[-2])
    traits_fp.unlink()


if __name__ == "__main__":
    main()
    log.info("Done!")

"""Extracts, cleans, and gets species mean trait values from TRY data."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.df_utils import filter_outliers
from src.utils.trait_utils import clean_species_name, get_trait_number_from_id


def _filter_if_specified(
    df: pd.DataFrame, trait_cols: list[str], quantile_range: tuple[float, float] | None
) -> pd.DataFrame:
    """Filter out outliers if specified."""
    if quantile_range is not None:
        return filter_outliers(df, cols=trait_cols, quantiles=quantile_range)
    return df


def _log_transform_long_tails(df: pd.DataFrame, keep: list[str] | None) -> pd.DataFrame:
    """Log-transform columns with long tails."""
    if keep is None:
        keep = []

    trait_cols = [col for col in df.columns if col.startswith("X")]

    long_tailed_cols = [
        col for col in trait_cols if int(get_trait_number_from_id(col)) not in keep
    ]
    # new_cols = [f"{col}_ln" for col in long_tailed_cols]

    if not long_tailed_cols:
        log.info("No long-tailed columns to log-transform.")
        return df

    log.info(f"Log-transforming long-tailed columns: {long_tailed_cols}")
    return df.assign(
        **{f"{col}_ln": (df[col].apply(np.log1p)) for col in long_tailed_cols}
    ).drop(columns=long_tailed_cols)


def standardize_trait_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize trait IDs between TRY5 and TRY6."""
    old_trait_cols = df.columns[df.columns.str.startswith("X")]
    new_trait_cols = [col.split(".")[1] for col in old_trait_cols if "." in col]
    return df.rename(columns=dict(zip(old_trait_cols, new_trait_cols)))


def main(cfg: ConfigBox = get_config()) -> None:
    """Extract, clean, and get species mean trait values from TRY data."""
    try_raw_dir = Path(cfg.raw_dir, cfg.trydb.raw.dir)
    try_prep_dir = Path(cfg.interim_dir, cfg.trydb.interim.dir)

    log.info("Extracting raw TRY traits data...")
    with zipfile.ZipFile(try_raw_dir / cfg.trydb.raw.zip, "r") as zip_ref:
        with zip_ref.open(cfg.trydb.raw.zipfile_csv) as zf:
            if cfg.trydb.raw.zipfile_csv.endswith(".zip"):
                with zipfile.ZipFile(zf, "r") as nested_zip_ref:
                    with nested_zip_ref.open(nested_zip_ref.namelist()[0]) as csvfile:
                        traits = pd.read_csv(csvfile, encoding="latin-1", index_col=0)
            else:
                traits = pd.read_csv(zf, encoding="latin-1", index_col=0)

    log.info("Filtering outliers and getting species mean trait values...")
    trait_cols = [col for col in traits.columns if col.startswith("X")]
    keep_cols = ["Species"] + trait_cols
    mean_filt_traits = (
        traits[keep_cols]
        .pipe(_filter_if_specified, trait_cols, cfg.trydb.interim.quantile_range)
        .pipe(standardize_trait_ids)
        .pipe(_log_transform_long_tails, cfg.trydb.raw.already_norm)
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


if __name__ == "__main__":
    main()
    log.info("Done!")

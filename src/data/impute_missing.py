"""Impute missing values in a dataset."""

from pathlib import Path

import pandas as pd
from box import ConfigBox
from verstack import NaNImputer

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import get_predict_fn


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in a dataset."""
    imputer = NaNImputer()
    df_imputed = imputer.impute(df)
    return df_imputed


def write_imputed_df(df: pd.DataFrame, cfg: ConfigBox) -> None:
    """Write the imputed dataset to disk."""
    out_fn = Path(
        cfg.interim_dir,
        cfg.eo_data.interim.dir,
        cfg.eo_data.imputed.dir,
        cfg.eo_data.imputed.filename,
    )

    df.to_parquet(out_fn, index=False)


def main(cfg: ConfigBox = get_config()) -> None:
    """Impute missing values in a dataset."""

    # Load the dataset
    log.info("Loading the dataset...")
    predict_df = pd.read_parquet(get_predict_fn(cfg))

    # Impute missing values
    log.info("Imputing missing values...")
    predict_imputed_df = impute_missing(predict_df)

    # Save the imputed dataset
    log.info("Saving the imputed dataset...")
    write_imputed_df(predict_imputed_df, cfg)

    log.info("Done!")


if __name__ == "__main__":
    main()

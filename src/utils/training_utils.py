"""Utility functions for model training."""

import numpy as np
import pandas as pd

from src.utils.dataset_utils import get_cv_splits_dir


def set_yx_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set the DataFrame index to "y" and "x"."""
    if not df.index.names == ["y", "x"]:
        return df.set_index(["y", "x"])
    return df


def assign_splits(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Assign the cross-validation splits to the DataFrame based on the label column."""
    splits = pd.read_parquet(get_cv_splits_dir() / f"{label_col}.parquet")

    return (
        df.pipe(set_yx_index)
        .merge(
            splits.pipe(set_yx_index), validate="m:1", right_index=True, left_index=True
        )
        .reset_index()
    )


def filter_trait_set(df: pd.DataFrame, trait_set: str) -> pd.DataFrame:
    """Filter the DataFrame based on the trait set."""
    # Check if "splot" and "gbif" are in split trait set
    if sorted(trait_set.split("_")) == ["gbif", "splot"]:
        # Remove duplicated rows in favor of "source" == "s"
        return df.sort_values(by="source", ascending=False).drop_duplicates(
            subset=["x", "y"], keep="first"
        )

    # Otherwise return the rows where "source" == "s" or "g" (depending
    # on trait_set)
    return df[df.source == trait_set[0]]


def assign_weights(
    df: pd.DataFrame,
    w_splot: int | float = 1.0,
    w_gbif: int | float = 0.08661,
) -> pd.DataFrame:
    """Assign weights to the DataFrame based on the source column. If only one source is
    present, assign a uniform weight of 1.0."""
    if df.source.unique().size == 1:
        return df.assign(weights=1.0)

    return df.assign(weights=np.where(df.source == "s", w_splot, w_gbif))

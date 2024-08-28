"""Utility functions for model training."""

import numpy as np
import pandas as pd


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

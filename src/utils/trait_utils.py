"""Utility functions for cleaning and processing trait data."""

import json
import re

import pandas as pd
from box import ConfigBox

from src.conf.conf import get_config


def genus_species_caps(col: pd.Series) -> pd.Series:
    """
    Converts the values in the given pandas Series to a format where the genus is
    capitalized and the species is lowercase.
    """
    col = col.str.title()

    return col.str.split().map(lambda x: x[0] + " " + x[1].lower())


def trim_species_name(col: pd.Series) -> pd.Series:
    """
    Trims the species name in the given column.
    """
    return col.str.extract("([A-Za-z]+ [A-Za-z]+)", expand=False)


def clean_species_name(
    df: pd.DataFrame, sp_col: str, new_sp_col: str | None = None
) -> pd.DataFrame:
    """
    Cleans a column containing species names by trimming them to the leading two words
    and ensuring they follow standard "Genus species" capitalization.
    """
    if new_sp_col is None:
        new_sp_col = sp_col

    return (
        df.assign(**{new_sp_col: trim_species_name(df[sp_col])})
        .dropna(subset=[new_sp_col])
        .assign(**{new_sp_col: lambda _df: genus_species_caps(_df[new_sp_col])})
    )


def filter_pft(df: pd.DataFrame, pft_set: str, pft_col: str = "pft") -> pd.DataFrame:
    """
    Filter the DataFrame based on the specified plant functional types (PFTs).

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        pft_set (str): The plant functional types to filter by, separated by underscores.
        pft_col (str, optional): The column name in the DataFrame that contains the PFTs.
            Defaults to "pft".

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        ValueError: If an invalid PFT designation is provided.
    """
    pfts = pft_set.split("_")
    if not any(pft in ["Shrub", "Tree", "Grass"] for pft in pfts):
        raise ValueError(f"Invalid PFT designation: {pft_set}")

    return df[df[pft_col].isin(pfts)]


def get_active_traits(cfg: ConfigBox = get_config()) -> list[str]:
    """Returns a list of full names of the active traits. E.g. ['X1_mean', 'X2_mean']"""
    y_cfg = cfg.datasets.Y
    return [f"X{i}_{y_cfg.trait_stats[y_cfg.trait_stat - 1]}" for i in y_cfg.traits]


def get_trait_number_from_id(trait_id: str) -> str:
    """Parses the trait number from a trait id string."""
    tnum = re.search(r"\d+", trait_id)
    if tnum is None:
        raise ValueError(f"Could not extract trait number from {trait_id}")
    return tnum.group()


def get_trait_name_from_id(trait_id: str, length: str = "short") -> tuple[str, str]:
    """Returns the name of a trait from its id as well as the unit of the trait."""
    with open(get_config().trait_mapping, encoding="utf-8") as f:
        mapping = json.load(f)

    tnum = get_trait_number_from_id(trait_id)

    if tnum not in mapping:
        raise ValueError(f"Trait number {tnum} not in mapping")

    if length not in mapping[tnum]:
        raise ValueError(f"Length {length} not in mapping for trait {trait_id}")

    return mapping[tnum][length], mapping[tnum]["unit"]

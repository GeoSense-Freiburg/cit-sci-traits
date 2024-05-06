"""Utility functions for cleaning and processing trait data."""

import pandas as pd


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

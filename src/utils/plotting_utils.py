import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from src.utils.trait_utils import get_trait_name_from_id


def set_font(font: str) -> None:
    """Set font for matplotlib plots."""
    if font not in ["FreeSans"]:
        raise ValueError("Font not supported yet.")

    if font == "FreeSans":
        font_path = "/usr/share/fonts/opentype/freefont/FreeSans.otf"
        font_bold_path = "/usr/share/fonts/opentype/freefont/FreeSansBold.otf"
        font_manager.fontManager.addfont(font_path)
        font_manager.fontManager.addfont(font_bold_path)
        plt.rcParams["font.family"] = "FreeSans"


def show_available_fonts() -> None:
    """Print all available fonts."""
    fonts = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    for font in fonts:
        print(font)


def add_human_readables(df: pd.DataFrame) -> pd.DataFrame:
    """Add human readable trait name and trait set abbreviations."""
    return df.pipe(add_trait_name).pipe(add_trait_set_abbr)


def add_trait_name(df: pd.DataFrame) -> pd.DataFrame:
    trait_id_to_name = {
        trait_id: get_trait_name_from_id(trait_id)[0]
        for trait_id in df.trait_id.unique()
    }
    return df.assign(trait_name=df.trait_id.map(trait_id_to_name))


def add_trait_set_abbr(df: pd.DataFrame) -> pd.DataFrame:
    trait_set_to_abbr = {"splot": "SCI", "splot_gbif": "COMB", "gbif": "CIT"}
    return df.assign(trait_set_abbr=df.trait_set.map(trait_set_to_abbr))

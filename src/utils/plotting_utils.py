import matplotlib.pyplot as plt
from matplotlib import font_manager


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

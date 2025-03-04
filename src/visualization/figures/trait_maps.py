from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
from rasterio.enums import Resampling

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import get_predict_dir
from src.utils.plotting_utils import set_font
from src.utils.raster_utils import open_raster
from src.utils.trait_utils import get_trait_number_from_id, load_trait_mapping

cfg = get_config()
# traits_of_interest = ["4", "3117", "14", "3106", "26", "3113"]
aoi_top_bbox = (8, 32, 70, -35)  # N Europe -> S Africa
aoi_bot_bbox = (-86, -62, 49, -56)  # N America -> S America
SAVE = True


def get_traits_of_interest() -> list[str]:
    res = (
        (
            pd.read_parquet(
                "results/all_results.parquet",
                columns=[
                    "trait_set",
                    "trait_id",
                    "pearsonr",
                    "transform",
                    "resolution",
                ],
            )
            .query(
                "trait_set == 'splot_gbif' and resolution == '1km' and "
                "transform == 'power'"
            )
            .sort_values(by="pearsonr", ascending=False)
        )
        .trait_id.head(6)
        .tolist()
    )
    return [get_trait_number_from_id(trait_id) for trait_id in res]


traits_of_interest = get_traits_of_interest()


def load_aois() -> tuple[list[Path], tuple[list[xr.DataArray], list[xr.DataArray]]]:
    # fns = [
    #     list((f / "splot_gbif").glob("*.tif"))[0]
    #     for f in get_predict_dir().iterdir()
    #     if f.is_dir() and get_trait_number_from_id(f.stem) in traits_of_interest
    # ]

    fns = []
    for trait_num in traits_of_interest:
        for d in get_predict_dir().iterdir():
            if d.is_dir() and get_trait_number_from_id(d.stem) == trait_num:
                fns.append(list((d / "splot_gbif").glob("*.tif"))[0])

    log.info("Loading and reprojecting prediction maps...")
    reproj = [
        open_raster(r)
        .sel(band=1)
        .rio.reproject(
            dst_crs="EPSG:4326", resolution=0.04, resampling=Resampling.average
        )
        for r in fns
    ]

    log.info("Loading and cropping AOIs...")
    aois_afr = [
        r.sel(x=slice(*aoi_top_bbox[:2]), y=slice(*aoi_top_bbox[2:])) for r in reproj
    ]
    log.info("...")
    aois_amr = [
        r.sel(x=slice(*aoi_bot_bbox[:2]), y=slice(*aoi_bot_bbox[2:])) for r in reproj
    ]

    return fns, (aois_afr, aois_amr)


def main():
    set_font("FreeSans")
    fns, (aois_top, aois_bot) = load_aois()
    build_plot(aois_top, aois_bot, fns)
    if SAVE:
        plt.savefig("results/figures/trait-maps.png", dpi=300, bbox_inches="tight")


def build_plot(
    aois_top: list[xr.DataArray], aois_bot: list[xr.DataArray], fns: list[Path]
) -> None:
    cmap = sns.color_palette("mako", as_cmap=True)
    mapping = load_trait_mapping()
    nrows = 2
    ncols = len(fns)
    width = 8.3 * 2
    height = 11.7 * 2 / nrows

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width, height),
        dpi=300,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    for i, (aoi, ax, fn) in enumerate(zip(aois_top + aois_bot, axes.ravel(), fns * 2)):
        min_val = aoi.quantile(0.02).values
        max_val = aoi.quantile(0.98).values
        mapping_entry = mapping[get_trait_number_from_id(fn.stem)]

        left, *_, right = aoi.x
        top, *_, bottom = aoi.y
        extent = [left, right, bottom, top]
        ax.imshow(
            aoi,
            cmap=cmap,
            vmin=min_val,
            vmax=max_val,
            transform=ccrs.PlateCarree(),
            extent=extent,
        )

        ax.set_title("")
        ax.coastlines(linewidth=0.2, color="white")
        ax.set_facecolor("black")

        # Remove border from subplots
        ax.spines["geo"].set_visible(False)

        if i < ncols:
            ax.set_title(
                mapping[get_trait_number_from_id(fns[i].stem)]["short"],
                fontsize=8,
                y=1,
            )
        else:
            cax = ax.images[0]
            bbox = ax.get_position()
            # Calculate the colorbar's position
            padding_x = 0.01  # Padding on the left
            padding_y = 0.07  # Padding on the bottom

            cbar_width = bbox.width * 0.075
            cbar_height = bbox.height * 0.18

            cbar_x = bbox.x0 + padding_x  # Bottom-left corner with padding
            cbar_y = bbox.y0 + padding_y

            # Add a new axes for the colorbar
            cbar_ax = fig.add_axes((cbar_x, cbar_y, cbar_width, cbar_height))

            # Add the colorbar
            cb = fig.colorbar(cax, cax=cbar_ax, orientation="vertical", extend="both")

            cb.set_label(
                f"{mapping_entry['short']} [{mapping_entry['unit']}]",
                color="white",
                labelpad=2,
                fontsize=7,
            )
            cb.ax.yaxis.set_label_position("left")
            cb.ax.yaxis.set_ticks_position("right")
            cb.ax.tick_params(color="white", labelcolor="white", labelsize=6)
            for spine in cb.ax.spines.values():
                spine.set_edgecolor("white")  # Set the border color
                spine.set_linewidth(0.3)  # Optional: Set the border width

        plt.subplots_adjust(wspace=-0.86, hspace=0.01)


if __name__ == "__main__":
    main()

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import QuadMesh
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from rasterio.enums import Resampling

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import get_aoa_dir, get_cov_dir, get_final_fns
from src.utils.plotting_utils import add_human_readables, set_font
from src.utils.raster_utils import create_sample_raster, open_raster
from src.utils.trait_utils import get_trait_number_from_id
from src.visualization.figures.model_performance import (
    add_biome_names,
    trait_set_by_biome_box_plot,
)

cfg = get_config()
# traits_of_interest = ["4", "14", "50"]
SAVE = True
USE_TMP = True
DPI = 300
TRAIT_SET_ORDER = ["SCI", "COMB", "CIT"]
colors = ["#b0b257", "#66a9aa", "#b95fa1"]
tricolor_palette = sns.color_palette(colors)
tricolor_palette_cmap = LinearSegmentedColormap.from_list(
    "tricolor_palette", [colors[1], colors[0], colors[2]]
)
coarsen_factor = 1


def get_traits_of_interest() -> list[str]:
    best_traits = pd.read_parquet(
        "results/all_results.parquet",
        columns=["trait_set", "trait_id", "resolution", "pearsonr", "transform"],
    ).query("resolution == '1km' and pearsonr >= 0.5 and transform == 'power'")
    splot_unique = best_traits.query("trait_set == 'splot'").trait_id.unique()
    comb_unique = best_traits.query("trait_set == 'splot_gbif'").trait_id.unique()
    unique_traits = set(splot_unique).intersection(set(comb_unique))
    return [get_trait_number_from_id(trait_id) for trait_id in unique_traits]


traits_of_interest = get_traits_of_interest()


def main() -> None:
    data = load_or_generate_data()

    with sns.plotting_context("paper", font_scale=1):
        set_font("FreeSans")
        build_figure(data)

    if SAVE:
        plt.savefig(
            "results/figures/spatial-transfer.png", dpi=DPI, bbox_inches="tight"
        )

    # plt.show()


def load_or_generate_data():
    if USE_TMP:
        tmp_dir = Path("tmp", "spatial_transferability_fig")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fns = [
            tmp_dir / "splot_mean.tif",
            tmp_dir / "comb_mean.tif",
            tmp_dir / "both_aoa.tif",
            tmp_dir / "mean_cov_df.parquet",
            tmp_dir / "merged_aoa_results.parquet",
        ]
        data = {}
        if all(f.exists() for f in fns):
            log.info("All files exist, loading from disk")
            data["splot_cov"] = open_raster(tmp_dir / "splot_mean.tif")
            data["comb_cov"] = open_raster(tmp_dir / "comb_mean.tif")
            data["both_aoa"] = open_raster(tmp_dir / "both_aoa.tif")
            data["merged_aoa_results"] = pd.read_parquet(
                tmp_dir / "merged_aoa_results.parquet"
            )
            data["mean_cov_df"] = pd.read_parquet(tmp_dir / "mean_cov_df.parquet")
        else:
            data = load_and_process_data()
            data["splot_cov"].rio.to_raster(tmp_dir / "splot_mean.tif")
            data["comb_cov"].rio.to_raster(tmp_dir / "comb_mean.tif")
            data["both_aoa"].rio.to_raster(tmp_dir / "both_aoa.tif")
            data["merged_aoa_results"].to_parquet(
                tmp_dir / "merged_aoa_results.parquet"
            )
            data["mean_cov_df"].to_parquet(tmp_dir / "mean_cov_df.parquet")

        data["biome_stats"] = pd.read_parquet("results/all_biome_results.parquet")
        data["biome_stats"] = data["biome_stats"][
            data["biome_stats"]["trait_id"]
            .str.extract(r"(\d+)")
            .isin(traits_of_interest)
            .values
        ]
    else:
        data = load_and_process_data()

    return data


def load_and_process_data():
    # Using fewer traits this time due to errors in COV calculation during the TRY5-STG-001
    # run (this may have been due to the presence of negative values in the predictions)
    splot_cov, comb_cov = aggregate_covs()
    both_aoa = get_aoa_difference_map()

    filtered_aoa_results = load_filtered_aoa_results()
    merged_aoa_results = reshape_aoa_results_and_index_by_trait(filtered_aoa_results)
    extent = [-180, 180, 90, -60]
    if splot_cov.rio.crs.to_epsg() == 6933:
        # transform the extent from epsg 4326 to 6933 using pyproj
        extent = [-17367530.45, 17367530.45, 7324184.56, -7324184.56]

    if coarsen_factor > 1:
        splot_cov = coarsen_xarray(splot_cov, extent, coarsen_factor)
        comb_cov = coarsen_xarray(comb_cov, extent, coarsen_factor)
        both_aoa = coarsen_xarray(both_aoa, extent, coarsen_factor)

    mean_cov_df = pd.DataFrame(
        {
            "Mean coefficient of variation": splot_cov.values.flatten().tolist()
            + comb_cov.values.flatten().tolist(),
            "Trait source": ["SCI"] * len(splot_cov.values.flatten())
            + ["COMB"] * len(comb_cov.values.flatten()),
        }
    )

    biome_stats = pd.read_parquet("results/all_biome_results.parquet")
    # Drop biome stats where trait_id is not in traits_of_interest. trait_id values are
    # formatted like "X14_mean" and traits_of_interest are strings representing only the
    # number, like "14"
    biome_stats = biome_stats[
        biome_stats["trait_id"].str.extract(r"(\d+)").isin(traits_of_interest).values
    ]

    return {
        "mean_cov_df": mean_cov_df,
        "splot_cov": splot_cov,
        "comb_cov": comb_cov,
        "merged_aoa_results": merged_aoa_results,
        "both_aoa": both_aoa,
        "mean_cov_by_biome": mean_cov_by_biome,
        "biome_stats": biome_stats,
    }


def build_figure(data: dict) -> None:
    fig, axes = scaffold_figure(dpi=DPI)

    mean_cov_density(data["mean_cov_df"], axes["ax0_0"])

    mean_cov_by_biome(data["biome_stats"], ax=axes["ax0_1"])

    aoa_fracs_by_resolution(data["merged_aoa_results"], ax=axes["ax0_2"])

    cov_vmax = data["splot_cov"].quantile(0.99).values
    cmap = sns.color_palette("mako", as_cmap=True)

    global_mean_cov_splot(
        data["splot_cov"], ax=axes["ax1_0"], vmin=0, vmax=cov_vmax, cmap=cmap
    )

    im = global_mean_cov_comb(
        data["comb_cov"], ax=axes["ax1_1"], vmin=0, vmax=cov_vmax, cmap=cmap
    )

    plot_colorbar(fig=fig, ax=axes["ax1_2"], im=im)

    # Col: 1, Row: 3
    global_aoa_diff(
        data["both_aoa"],
        ax=axes["ax1_3"],
        map_cmap=tricolor_palette_cmap,
        legend_palette=tricolor_palette,
    )

    letter_size = 10
    x_offset_left_col = -0.15
    y_offset_left_col = 1.10

    x_offset_right_col = 0.01
    y_offset_right_col = 1.1
    add_subplot_letter(
        ax=axes["ax0_0"],
        letter_size=letter_size,
        x=x_offset_left_col,
        y=y_offset_left_col,
        letter="a",
    )
    add_subplot_letter(
        ax=axes["ax0_1"],
        letter_size=letter_size,
        x=x_offset_left_col,
        y=y_offset_left_col,
        letter="b",
    )
    add_subplot_letter(
        ax=axes["ax1_0"],
        letter_size=letter_size,
        x=x_offset_right_col,
        y=y_offset_right_col,
        letter="c",
    )
    add_subplot_letter(
        ax=axes["ax0_2"],
        letter_size=letter_size,
        x=x_offset_left_col,
        y=y_offset_left_col,
        letter="d",
    )
    add_subplot_letter(
        ax=axes["ax1_3"],
        letter_size=letter_size,
        x=x_offset_right_col,
        y=y_offset_right_col,
        letter="e",
    )


def scaffold_figure(dpi: int = 300) -> tuple[Figure, dict[str, Axes | GeoAxes]]:
    """
    Creates and returns a figure and a nested grid of axes for plotting.

    Args:
        dpi (int): Dots per inch for the figure. Default is 300.

    Returns:
        tuple[Figure, Sequence[Sequence[Axes]]]:
        A tuple containing the figure and a nested sequence of axes.
    """
    nrows = 1
    ncols = 2
    width = 15
    height = width * (2 / 3)
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    gs = gridspec.GridSpec(
        nrows, ncols, figure=fig, width_ratios=[0.8, 4], hspace=0, wspace=-0.5
    )
    # Create a nested GridSpec within the first column to add padding
    spacer = 0.4
    left_width = 1
    nested_gs_left = gridspec.GridSpecFromSubplotSpec(
        3,
        1,
        subplot_spec=gs[:, 0],
        height_ratios=[0.8, 1, 1],
        width_ratios=[left_width],
        hspace=spacer,
    )
    nested_gs_right = gridspec.GridSpecFromSubplotSpec(
        5,
        3,
        subplot_spec=gs[:, 1],
        height_ratios=[1, 1, 0.05, 0.1, 1],
        width_ratios=[1, 0.9, 1],
        hspace=0.3,
    )

    subplots = {
        "ax0_0": fig.add_subplot(nested_gs_left[0, 0]),
        "ax0_1": fig.add_subplot(nested_gs_left[1, 0]),
        "ax0_2": fig.add_subplot(nested_gs_left[2, 0]),
        "ax1_0": fig.add_subplot(nested_gs_right[0, :], projection=ccrs.EqualEarth()),
        "ax1_1": fig.add_subplot(nested_gs_right[1, :], projection=ccrs.EqualEarth()),
        "ax1_2": fig.add_subplot(nested_gs_right[2, 1], frameon=False),
        "ax1_3": fig.add_subplot(nested_gs_right[4, :], projection=ccrs.EqualEarth()),
    }

    return fig, subplots


def reshape_aoa_results_and_index_by_trait(
    filtered_aoa_results: pd.DataFrame,
) -> pd.DataFrame:
    resolutions = filtered_aoa_results.resolution.unique()
    aoa_data = pd.DataFrame()

    for res in resolutions:
        res_res = filtered_aoa_results.query(f"resolution == '{res}'")
        splot = res_res.query("trait_set == 'splot'").set_index("trait_id")["aoa"]
        comb = res_res.query("trait_set == 'splot_gbif'").set_index("trait_id")["aoa"]

        merged = pd.merge(
            splot, comb, left_index=True, right_index=True, suffixes=("_splot", "_comb")
        )
        merged["resolution"] = res
        aoa_data = pd.concat([aoa_data, merged], ignore_index=True).sort_values(
            "resolution", ascending=True
        )

    # resolution_map = {
    #     "001": "0.01",
    #     "02": "0.2",
    #     "05": "0.5",
    #     "1": "1",
    #     "2": "2",
    # }

    # aoa_data["resolution"] = aoa_data["resolution"].map(resolution_map)
    return aoa_data


def load_filtered_aoa_results():
    aoa_results = pd.read_parquet("results/all_aoa.parquet").astype({"resolution": str})
    filt_results = aoa_results.query(
        "transform == 'power' and "
        "(trait_set == 'splot_gbif' or trait_set == 'splot') and "
        "resolution.str.contains('km')"
    )

    resolutions = filt_results.resolution.unique()

    # Filter out models where we don't have both a splot and splot_gbif model
    resolution_dfs = []
    for res in resolutions:
        res_results = filt_results.query(f"resolution == '{res}'")
        splot_traits = res_results.query("trait_set == 'splot'").trait_id.unique()
        comb_traits = res_results.query("trait_set == 'splot_gbif'").trait_id.unique()

        # Identify the traits that are in both sets
        common_traits = set(splot_traits).intersection(set(comb_traits))

        # Filter out the models that don't have both trait sets
        res_results = res_results[res_results["trait_id"].isin(common_traits)]
        resolution_dfs.append(res_results)

    return pd.concat(resolution_dfs)


def coarsen_xarray(
    da: xr.DataArray, extent: Sequence[int | float], factor: int
) -> xr.DataArray:
    return (
        da.sel(x=slice(*extent[:2]), y=slice(*extent[2:]))
        .coarsen(x=factor, y=factor, boundary="trim")
        .mean()
    )


def get_aoa_difference_map() -> xr.DataArray:
    splot_aoa_fns, comb_final_fns = get_aoa_fns()
    splot_aoas, comb_aoas = load_aoas(splot_aoa_fns, comb_final_fns)
    splot_aoa, comb_aoa = aggregate_aoas(
        splot_aoa_fns, splot_aoas, comb_final_fns, comb_aoas
    )
    splot_aoa, comb_aoa = reduce_aoas_by_mode(splot_aoa, comb_aoa)
    both_aoa = combine_aoas_with_exclusion(splot_aoa, comb_aoa)
    return both_aoa


def combine_aoas_with_exclusion(
    splot_aoa: xr.DataArray, comb_aoa: xr.DataArray
) -> xr.DataArray:
    # 1: Both outside AOA
    # 2: Only sPlot outside AOA
    # 3: Only COMB outside AOA
    both_aoa = xr.where(
        (splot_aoa == 1) & (comb_aoa == 1),
        1,
        xr.where(
            (splot_aoa == 1) & (comb_aoa == 0),
            2,
            xr.where((splot_aoa == 0) & (comb_aoa == 1), 3, np.nan),
        ),
    )

    return both_aoa


def reduce_aoas_by_mode(splot_aoa, comb_aoa):
    majority_thresh = len(traits_of_interest) / 2
    splot_aoa = xr.where(splot_aoa >= majority_thresh, 1, splot_aoa)
    comb_aoa = xr.where(comb_aoa >= majority_thresh, 1, comb_aoa)
    return splot_aoa, comb_aoa


def load_aoas(splot_aoa_fns, comb_final_fns):
    splot_aoas = [open_raster(f, masked=True).sel(band=2) for f in splot_aoa_fns]
    comb_aoas = [open_raster(f).sel(band=3) for f in comb_final_fns]
    return splot_aoas, comb_aoas


def aggregate_aoas(
    splot_aoa_fns: Sequence[Path],
    splot_aoa_das: Sequence[xr.DataArray],
    comb_aoa_fns: Sequence[Path],
    comb_aoa_das: Sequence[xr.DataArray],
) -> tuple[xr.DataArray, xr.DataArray]:
    # Fix incorrect nodata encoding by replacing non-binary values with NaN
    # splot_aoa_das = (
    #     xr.where((da == 0) | (da == 1), da, np.nan) for da in splot_aoa_das
    # )

    splot_aoas_ds = xr.Dataset(
        {
            file_path.parents[1].name: da
            for da, file_path in zip(splot_aoa_das, splot_aoa_fns)
        }
    )
    comb_aoas_ds = xr.Dataset(
        {  # Takes a name like 'X50_mean_Shrub_Tree_Grass_1km" and returns 'X50_mean'
            "_".join(f.name.split("_", 2)[:2]): da
            for da, f in zip(comb_aoa_das, comb_aoa_fns)
        }
    )

    splot_aoa_da = splot_aoas_ds.to_dataarray(dim="band").sum(dim="band", skipna=False)
    comb_aoa_da = comb_aoas_ds.to_dataarray(dim="band").sum(dim="band", skipna=False)
    return splot_aoa_da, comb_aoa_da


def get_aoa_fns():
    splot_aoa_fns = [
        list((f / "splot").glob("*.tif"))[0]
        for f in sorted(list(get_aoa_dir().iterdir()))
        if f.is_dir() and get_trait_number_from_id(f.stem) in traits_of_interest
    ]

    comb_final_fns = [
        f
        for f in get_final_fns()
        if get_trait_number_from_id(f.stem) in traits_of_interest
    ]

    return splot_aoa_fns, comb_final_fns


def aggregate_covs():
    splot_fns, comb_fns = get_cov_filepaths()

    splot_covs_and_ids = [
        (open_raster(f).sel(band=1), f.parents[1].name) for f in splot_fns
    ]
    comb_covs_and_ids = [
        (open_raster(f).sel(band=1), f.parents[1].name) for f in comb_fns
    ]

    splot_covs = xr.Dataset({t_id: da for da, t_id in splot_covs_and_ids})
    comb_covs = xr.Dataset({t_id: da for da, t_id in comb_covs_and_ids})

    splot_mean = splot_covs.to_array(dim="mean").mean(dim="mean")
    comb_mean = comb_covs.to_array(dim="mean").mean(dim="mean")
    return splot_mean, comb_mean


def get_cov_filepaths():
    splot_fns = [
        list((f / "splot").glob("*.tif"))[0]
        for f in sorted(list(get_cov_dir().iterdir()))
        if f.is_dir() and get_trait_number_from_id(f.stem) in traits_of_interest
    ]

    comb_fns = [
        list((f / "splot_gbif").glob("*.tif"))[0]
        for f in sorted(list(get_cov_dir().iterdir()))
        if f.is_dir() and get_trait_number_from_id(f.stem) in traits_of_interest
    ]

    return splot_fns, comb_fns


def mean_cov_density(df: pd.DataFrame, ax: Axes) -> None:
    pdf_cmap = sns.color_palette("crest", as_cmap=True)
    desat = 1
    color_start = sns.desaturate(pdf_cmap(0.1), desat)
    color_end = sns.desaturate(pdf_cmap(0.9), desat)

    sns.kdeplot(
        data=df,
        x="Mean coefficient of variation",
        hue="Trait source",
        fill=True,
        multiple="layer",
        # palette=sns.color_palette("crest", 2, desat=0.75),
        palette=colors[:2],
        alpha=0.5,
        linewidth=0,
        ax=ax,
        log_scale=(True, False),
    )

    legend = ax.get_legend()
    legend.set_frame_on(False)
    legend.set_title("")
    ax.set_xlabel("Mean coefficient of variation", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")

    sns.despine(ax=ax)


def mean_cov_by_biome(biome_stats: pd.DataFrame, ax: Axes) -> None:
    biome_stats = (
        biome_stats.query("trait_set != 'gbif'")
        .pipe(add_human_readables)
        .pipe(add_biome_names)[["biome_name", "trait_set_abbr", "mean_cov"]]
    )

    trait_set_by_biome_box_plot(
        df=biome_stats,
        ax=ax,
        metric_col="mean_cov",
        metric_label="Mean coefficient of variation",
        order=TRAIT_SET_ORDER[:2],
        palette=tricolor_palette[:2],
        fliersize=0,
        max_x=0.2,
    )
    sns.despine(ax=ax)


def aoa_fracs_by_resolution(aoa_data: pd.DataFrame, ax: Axes) -> None:
    def _sort_key(resolution: str) -> int:
        return int(resolution.replace("km", ""))

    palette = sns.color_palette(
        "husl", n_colors=aoa_data.resolution.nunique(), desat=0.5
    )
    hue_order = ["1km", "22km", "55km", "111km", "222km"]
    hue_order = [1, 22, 55, 111, 222]
    aoa_data = aoa_data.sort_values(
        by="resolution", key=lambda x: x.map(_sort_key)
    ).assign(
        aoa_comb=lambda _df: _df.aoa_comb * 100,
        aoa_splot=lambda _df: _df.aoa_splot * 100,
        resolution=lambda _df: _df.resolution.map(lambda x: int(x.replace("km", ""))),
    )

    # Convert aoa_comb and aoa_splot columns from fractions to percent when plotting

    # aoa_data = aoa_data.sort_values("resolution")
    sns.scatterplot(
        data=aoa_data,
        x="aoa_comb",
        y="aoa_splot",
        hue="resolution",
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        s=12,
        linewidth=0.3,
    )

    # ax.set_ylim(1.01)
    # Set legend title to "Resolution [°]"
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles[0:],
        labels=labels[0:],
        title="Resolution [km]",
        title_fontproperties={"style": "italic"},
        frameon=False,
        loc="best",
    )

    # include a 1:1 line
    p1 = min(aoa_data["aoa_splot"].min(), aoa_data["aoa_comb"].min())
    p2 = max(aoa_data["aoa_splot"].max(), aoa_data["aoa_comb"].max())
    ax.plot([p1, p2], [p1, p2], color="black", linestyle="--", alpha=0.5, lw=0.7)

    ax.set_xlabel("% within AOA (COMB)", fontweight="bold")
    ax.set_ylabel("% within AOA (SCI)", fontweight="bold")

    # Round ax0_2 y ticks to 2 decimal places
    # ax0_2.set_yticklabels([f"{tick:.2f}" for tick in ax0_2.get_yticks()])

    # Ensure y-ticks are the same as x-ticks
    # ax0_2.set_xlim(0.82, 1.01)
    ax.set_yticks(ax.get_xticks())
    xmin = ax.get_xlim()[0]
    ax.set_ylim(bottom=xmin, top=101)
    ax.set_xticks(ax.get_yticks())
    ax.set_xlim(left=xmin, right=101)
    # Get x min
    # ax0_2.set_yticklabels(ax0_2.get_xticklabels())
    # stop yticks after 1

    # fig.suptitle(
    #     "AOA comparison between sPlot-only\nand Combined models ($R^2 \\geq 0.2$)", y=1.05
    # )
    sns.despine(ax=ax)


def global_mean_cov_splot(
    da: xr.DataArray, ax: GeoAxes, cmap: Colormap, vmin: int | float, vmax: int | float
):
    plot_global_map(da=da, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Vegetation surveys only (SCI)", y=0.99)


def global_mean_cov_comb(
    da: xr.DataArray, ax: GeoAxes, cmap: Colormap, vmin: int | float, vmax: int | float
) -> QuadMesh:
    im = plot_global_map(da=da, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Citizen science + vegetation surveys (COMB)", y=0.99)
    return im


def plot_global_map(
    da: xr.DataArray,
    ax: GeoAxes,
    cmap: Colormap,
    vmin: int | float | None = None,
    vmax: int | float | None = None,
    resampling: Resampling = Resampling.average,
) -> QuadMesh:
    # Reproject to WGS84
    sample_raster = create_sample_raster(
        extent=[-180, -60, 180, 90], resolution=0.1, crs="EPSG:4326"
    )
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:6933", inplace=True)

    # da = da.rio.reproject_match(sample_raster)
    da = da.rio.write_nodata(np.nan)
    da_reproj = da.rio.reproject(
        dst_crs="EPSG:8857", resolution=10000, resampling=resampling
    )

    # map_extent = (-180, 180, -60, 90)

    xx, yy = np.meshgrid(da_reproj.x, da_reproj.y)
    im = ax.pcolormesh(
        xx,
        yy,
        da_reproj.values.squeeze(),
        cmap=cmap,
        transform=ccrs.EqualEarth(),
        # extent=map_extent,
        # resample=False,
        # aspect="auto",
        # aspect=1.4,
        shading="auto",
        rasterized=True,
        vmin=vmin,
        vmax=vmax,
    )

    # ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.2, color="black")
    # ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)
        spine.set_edgecolor("grey")

    return im


def global_aoa_diff(
    da: xr.DataArray, ax: GeoAxes, map_cmap: Colormap, legend_palette: Any
) -> None:
    plot_global_map(da, ax, map_cmap, resampling=Resampling.mode)
    legend_labels = ["SCI outside", "COMB outside", "Both outside"]
    # legend_colors = [legend_palette(i) for i in range(3)]
    legend_patches = [
        Patch(color=color, label=label)
        for color, label in zip(legend_palette, legend_labels)
    ]

    # Reorder legend patches such that the first item is the last item
    # legend_patches = legend_patches[1:] + legend_patches[:1]

    ax.legend(handles=legend_patches, loc="lower left", frameon=False)
    # s.set_edgecolor("grey")
    ax.set_title(
        "Differences in median area of applicability (AOA)", y=-0.2, fontweight="bold"
    )


def plot_colorbar(fig: Figure, ax: Axes, im: ScalarMappable):
    # # Remove ticks
    ax.set_xticks([])

    cbar = fig.colorbar(
        im,
        cax=ax,
        location="bottom",
        orientation="horizontal",
        # pad=1,
        extend="max",
        use_gridspec=True,
        # label="Mean coefficient of variation",
        panchor=(1, 0.5),
    )
    cbar.set_label("Mean coefficient of variation", fontweight="bold")

    for s in cbar.ax.spines.values():
        s.set_linewidth(0.2)


def add_subplot_letter(
    ax: Axes | GeoAxes, letter_size: int, x: float, y: float, letter: str
):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=letter_size,
        verticalalignment="top",
        fontweight="bold",
    )


if __name__ == "__main__":
    main()

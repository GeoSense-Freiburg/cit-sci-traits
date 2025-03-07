import argparse
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import get_biome_mapping
from src.utils.plotting_utils import add_human_readables, set_font
from src.visualization.model_assessment import plot_splot_correlations

TRAIT_SET_ORDER = ["SCI", "COMB", "CIT"]
tricolor_palette = sns.color_palette(["#b0b257", "#66a9aa", "#b95fa1"])
CFG = get_config()


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model performance figure.")
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="./results/figures/model-performance.png",
        help="Output file path.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    set_font("FreeSans")

    log.info("Compiling results...")
    all_res, biome_res = compile_results()

    log.info("Building figure...")
    with sns.plotting_context("paper", 1.5):
        build_figure(all_res, biome_res)

    if args is not None:
        log.info("Saving figure...")
        plt.savefig(args.out_path, dpi=300, bbox_inches="tight")

    plt.show()


def compile_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    drop_calls_all_res = ["automl", "model_arch"]
    drop_cols_both = [
        "trait_set",
        "run_id",
        "pearsonr_wt",
        "root_mean_squared_error",
        "mean_squared_error",
        "mean_absolute_error",
        "median_absolute_error",
    ]

    # Only keep traits listed in params
    keep_traits = [f"X{t}" for t in CFG.datasets.Y.traits]  # noqa: F841

    all_results = (
        pd.read_parquet("results/all_results.parquet")
        .assign(base_trait_id=lambda df: df.trait_id.str.split("_").str[0])
        .query("base_trait_id in @keep_traits")
        .pipe(add_human_readables)
        .drop(columns=drop_cols_both + drop_calls_all_res + ["base_trait_id"])
        .query("transform == 'power'")
    )
    biome_results = (
        pd.read_parquet("results/all_biome_results.parquet")
        .assign(base_trait_id=lambda df: df.trait_id.str.split("_").str[0])
        .query("base_trait_id in @keep_traits")
        .pipe(add_human_readables)
        .pipe(add_biome_names)
        .drop(columns=drop_cols_both + ["base_trait_id"])
        .query("transform == 'power'")
    )

    biome_results.to_parquet("tmp/biomes.parquet")

    return all_results, biome_results


def build_figure(all_res: pd.DataFrame, biome_res: pd.DataFrame) -> Figure:
    fig, axes = scaffold_figure()
    keep_cols = ["trait_name", "trait_set_abbr"]

    # Col: 0, Row: 0
    r_by_trait_set(
        df=all_res.query("resolution == '1km'")[keep_cols + ["pearsonr"]], ax=axes[0][0]
    )

    # Col: 0, Row 1
    nrmse_by_trait_set(
        df=all_res.query("resolution == '1km'")[
            keep_cols + ["norm_root_mean_squared_error"]
        ],
        ax=axes[0][1],
    )

    # Col: 0, Row: 2
    nrmse_by_biome(
        df=biome_res.query("resolution == '1km'")[
            keep_cols + ["biome_name", "norm_root_mean_squared_error"]
        ],
        ax=axes[0][2],
    )

    # Col: 1, Row: All
    r_by_resolution(
        df=all_res[all_res.resolution.str.contains("km")][
            keep_cols + ["trait_id", "pft", "transform", "resolution", "pearsonr"]
        ],
        ax=axes[1],
        trait_set="COMB",
    )

    letter_size = 10
    x_offset_left_col = -0.02
    y_offset_left_col = 1.10

    x_offset_right_col = -0.02
    y_offset_right_col = 1.03
    add_subplot_letter(
        ax=axes[0][0],
        letter_size=letter_size,
        x=x_offset_left_col,
        y=y_offset_left_col,
        letter="a",
    )
    add_subplot_letter(
        ax=axes[0][1],
        letter_size=letter_size,
        x=x_offset_left_col,
        y=y_offset_left_col - 0.03,
        letter="b",
    )
    add_subplot_letter(
        ax=axes[0][2],
        letter_size=letter_size,
        x=x_offset_left_col,
        y=y_offset_left_col - 0.05,
        letter="c",
    )
    add_subplot_letter(
        ax=axes[1],
        letter_size=letter_size,
        x=x_offset_right_col,
        y=y_offset_right_col,
        letter="d",
    )

    return fig


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


def scaffold_figure(dpi: int = 100) -> tuple[Figure, tuple[list[Axes], Axes]]:
    nrows, ncols = 1, 2
    left_col_nrows = 3

    height = left_col_nrows * 5
    width = ncols * 5

    fig = plt.figure(figsize=(width, height), dpi=dpi)
    gs = GridSpec(nrows, ncols)
    violin_plot_height = 0.7
    nested_gs_left = GridSpecFromSubplotSpec(
        left_col_nrows,
        1,
        subplot_spec=gs[0, 0],
        height_ratios=[violin_plot_height, violin_plot_height, 1],
    )

    ax0_0 = plt.subplot(nested_gs_left[0])
    ax0_1 = plt.subplot(nested_gs_left[1])
    ax0_2 = plt.subplot(nested_gs_left[2])
    ax1 = plt.subplot(gs[1])

    return fig, ([ax0_0, ax0_1, ax0_2], ax1)


def r_by_trait_set(df: pd.DataFrame, ax: Axes) -> Axes:
    return trait_set_comparison_violin_plot(
        df,
        ax,
        metric_col="pearsonr",
        metric_label="Pearson's $r$",
        order=TRAIT_SET_ORDER,
    )


def nrmse_by_trait_set(df: pd.DataFrame, ax: Axes) -> Axes:
    return trait_set_comparison_violin_plot(
        df,
        ax,
        metric_col="norm_root_mean_squared_error",
        metric_label="$nRMSE$",
        order=TRAIT_SET_ORDER,
    )


def nrmse_by_biome(df: pd.DataFrame, ax: Axes) -> Axes:
    return trait_set_by_biome_box_plot(
        df,
        ax,
        metric_col="norm_root_mean_squared_error",
        metric_label="$nRMSE$",
        order=TRAIT_SET_ORDER,
        max_x=0.38,
    )


def r_by_resolution(df: pd.DataFrame, ax: Axes, trait_set: str) -> Axes:
    return plot_splot_correlations(
        df,
        "Shrub_Tree_Grass",
        [trait_set],
        trait_set_ids_col="trait_set_abbr",
        axes=[ax],
    )[0]


def trait_set_comparison_violin_plot(
    df: pd.DataFrame, ax: Axes, metric_col: str, metric_label: str, order: list[str]
) -> Axes:
    df = (
        df.copy()
        .rename(columns={metric_col: metric_label})
        .sort_values(by="trait_set_abbr", ascending=False)
    )

    ax = sns.violinplot(
        x="trait_set_abbr",
        y=metric_label,
        data=df,
        ax=ax,
        inner="quart",
        linewidth=1,
        order=order,
        cut=0,
        saturation=0.5,
        hue="trait_set_abbr",
        palette=tricolor_palette,
        alpha=0.05,
        inner_kws={"linewidth": 0.2, "alpha": 1},
    )

    sns.stripplot(
        x="trait_set_abbr",
        y=metric_label,
        data=df,
        ax=ax,
        palette=tricolor_palette,
        hue="trait_set_abbr",
        order=order,
        size=4,
        jitter=0.15,
        linewidth=0,
    )

    ax = annotate_top_and_bottom_points(ax, df, metric_label, order)

    ax.set_ylabel(metric_label, labelpad=15)
    ax.set_xlabel("")
    ax.legend().remove()
    sns.despine()

    return ax


def trait_set_by_biome_box_plot(
    df: pd.DataFrame,
    ax: Axes,
    metric_col: str,
    metric_label: str,
    order: list[str],
    palette: Any | None = tricolor_palette,
    max_x: float | None = None,
    fliersize: int = 0,
) -> Axes:
    ax = sns.boxplot(
        data=df,
        x=metric_col,
        y="biome_name",
        ax=ax,
        hue="trait_set_abbr",
        hue_order=order,
        palette=palette,
        dodge=True,
        fliersize=fliersize,
        linewidth=0.5,
    )
    if max_x is not None:
        ax.set_xlim(right=max_x)

    ax.set_xlabel(metric_label, fontweight="bold")
    ax.set_ylabel("Biome", fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    for patch in ax.get_legend().get_patches():
        patch.set_edgecolor("none")
    ax.legend(handles, labels, title=None, loc="best", frameon=False)
    return ax


def annotate_top_and_bottom_points(
    ax: Axes, df: pd.DataFrame, label: str, plot_order: list[str]
) -> Axes:
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.set_ylim(ax.get_ylim()[0] - (0.1 * y_range), ax.get_ylim()[1] + (0.1 * y_range))

    for i, trait_set in enumerate(plot_order):
        top_trait, bottom_trait = [
            split_trait_name(trait_name)
            for trait_name in get_marginal_trait_names(df, trait_set, label)
        ]

        offsets = ax.collections[3 + i].get_offsets()
        top_x = offsets[offsets[:, 1].argsort()][-1][0]
        top_y = offsets[offsets[:, 1].argsort()][-1][1] + (0.01 * y_range)
        bottom_x = offsets[offsets[:, 1].argsort()][0][0]
        bottom_y = offsets[offsets[:, 1].argsort()][0][1] - (0.01 * y_range)

        # Draw a small line diagonally at a random angle upwards
        top_angle = 90
        bottom_angle = 270
        length = 0.03 * y_range  # Length of the line
        dx_top = length * np.cos(np.radians(top_angle))
        dy_top = length * np.sin(np.radians(top_angle))
        dx_bottom = length * np.cos(np.radians(bottom_angle))
        dy_bottom = length * np.sin(np.radians(bottom_angle))

        ### Top annotation
        ax.annotate(
            "",
            xy=(top_x, top_y),
            xytext=(top_x + dx_top, top_y + dy_top),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
        )

        ax.annotate(
            top_trait,
            (top_x + dx_top, top_y + dy_top),
            textcoords="offset points",
            xytext=(0, 2),
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            annotation_clip=False,
        )

        ### Bottom annotation
        ax.annotate(
            "",
            xy=(bottom_x, bottom_y),
            xytext=(bottom_x + dx_bottom, bottom_y + dy_bottom),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
        )

        ax.annotate(
            bottom_trait,
            (bottom_x + dx_bottom, bottom_y + dy_bottom),
            textcoords="offset points",
            xytext=(0, -2),
            ha="center",
            va="top",
            fontsize=8,
            color="black",
            annotation_clip=False,
        )

    return ax


def get_marginal_trait_names(
    df: pd.DataFrame, trait_set: str, label: str
) -> tuple[str, str]:
    top = (
        df.query(f"trait_set_abbr == '{trait_set}'")
        .sort_values(by=label, ascending=False)
        .iloc[0]
        .trait_name
    )
    bottom = (
        df.query(f"trait_set_abbr == '{trait_set}'")
        .sort_values(by=label, ascending=True)
        .iloc[0]
        .trait_name
    )
    return top, bottom


def split_trait_name(trait_name: str) -> str:
    """Splits a trait name to two lines if it is too long."""
    if len(trait_name) > 10:
        # if >= 3 words, split on the second space, otherwise split on the first space
        if len(trait_name.split()) >= 3:
            split_idx = trait_name.find(" ", trait_name.find(" ") + 1)
        else:
            split_idx = trait_name.find(" ")
        trait_name = trait_name[:split_idx] + "\n" + trait_name[split_idx + 1 :]
    return trait_name


def add_biome_names(df: pd.DataFrame) -> pd.DataFrame:
    biome_mapping = {int(k): v for k, v in get_biome_mapping().items()}
    return df.pipe(lambda _df: _df.assign(biome_name=_df.biome.map(biome_mapping)))


if __name__ == "__main__":
    main(cli())

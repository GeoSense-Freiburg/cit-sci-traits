from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.axes import Axes

from src.utils.trait_utils import get_trait_name_from_id


def plot_observed_vs_predicted(
    ax: plt.Axes,
    observed: pd.Series,
    predicted: pd.Series,
    name: str,
    log: bool = False,
    density: bool = False,
    show_r: bool = True,
    manual_r: float | None = None,
    r_weighted: bool = False,
):
    """Plot observed vs. predicted values."""

    p1 = min(predicted.min(), observed.min())
    p2 = max(predicted.max(), observed.max())

    cmap = sns.cubehelix_palette(start=0.5, rot=-0.75, reverse=True, as_cmap=True)  # type: ignore
    if density:
        sns.kdeplot(x=predicted, y=observed, ax=ax, cmap=cmap, fill=True, thresh=0.0075)
    else:
        sns.scatterplot(x=predicted, y=observed, ax=ax, s=1)

    # Fit a regression line for observed vs. predicted values, plot the regression
    # line so that it spans the entire plot, and print the correlation coefficient
    m, b = np.polyfit(predicted, observed, 1)
    reg_line = [m * p1 + b, m * p2 + b]

    if log:
        ax.loglog([p1, p2], [p1, p2], color="black", ls="-.", lw=0.5, alpha=0.5)
        ax.loglog([p1, p2], reg_line, color="red", lw=0.5)
    else:
        ax.plot([p1, p2], [p1, p2], color="black", ls="-.", lw=0.5, alpha=0.5)
        ax.plot([p1, p2], reg_line, color="red", lw=0.5)

    # make sure lines are positioned on top of kdeplot
    ax.set_zorder(1)

    buffer_color = "#e9e9f1"

    if show_r:
        rval = np.corrcoef(predicted, observed)[0, 1] if manual_r is None else manual_r
        ax.text(
            0.05,
            0.95,
            f"$r${' (weighted)' if r_weighted else ''}= {rval:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": buffer_color, "edgecolor": buffer_color, "pad": 0.5},
        )

    ax.text(
        0.05,
        0.90,
        f"n = {len(predicted):,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": buffer_color, "edgecolor": buffer_color, "pad": 0.5},
    )

    # include legend items for the reg_line and the 1-to-1 line
    ax.legend(
        [
            ax.get_lines()[0],
            ax.get_lines()[1],
        ],
        ["1-to-1", "Regression"],
        loc="lower right",
        frameon=False,
    )

    # set informative axes and title
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(name)

    return ax


def plot_splot_correlations(
    df: pd.DataFrame,
    pft: str,
    trait_set_ids: list[str],
    trait_set_ids_col: str = "trait_set_id",
    axes: list[Axes] | None = None,
    unit: str = "km",
    out_path: Path | None = None,
):
    """Plot sPlot correlations for GBIF and sPlot extrapolations for the given PFT"""
    if axes is not None and len(axes) != len(trait_set_ids):
        raise ValueError("The number of axes must be equal to the number of trait sets")

    if unit not in ["km", "deg"]:
        raise ValueError("unit must be either 'km' or 'deg'")
    if unit == "km":
        df = df.query(
            f"pft == '{pft}' and transform == 'power' and resolution.str.endswith('km')"
        )
    else:
        df = df.query(
            f"pft == '{pft}' and transform == 'none' and resolution != '1km'"
        )[["trait_id", trait_set_ids_col, "pearsonr", "resolution"]]

    traits = df.trait_id.unique()

    resolution_labels = ["1 km", "22 km", "55 km", "111 km", "222km"]
    resolutions = ["1km", "22km", "55km", "111km", "222km"]
    if unit == "deg":
        resolution_labels = ["0.01", "0.2", "0.5", "1", "2"]
        resolutions = ["001", "02", "05", "1", "2"]

    # Figure directory
    ncols = 1
    if axes is None or len(axes) > 1:
        nrows = 1
        ncols = len(trait_set_ids)
        _, axes = plt.subplots(
            nrows, ncols=ncols, figsize=(6.66 * ncols, 17 * nrows), dpi=300
        )
        if ncols > 1:
            axes = axes.flatten()

    Y_LIM = (0, 0.82)

    for ax, trait_set_id in zip(axes, trait_set_ids):
        text_x = 0.98

        # Define colors
        # colors = plt.cm.Paired(np.linspace(0, 1, len(stg.index.get_level_values(0).unique())))
        # use sns instead (e.g. sns.hls_palette(h=.5))
        colors = sns.color_palette(n_colors=len(traits), desat=0.75)

        x_positions = [text_x] * len(traits)
        y_positions = []
        labels = []
        label_colors = []

        # Loop over each trait
        for color, trait in zip(colors, traits):
            # Select data for the current trait
            trait_data = df.query(f"trait_id == '{trait}'")
            trait_short = get_trait_name_from_id(trait)[0]

            # Plot sPlot data with solid line and circular markers
            # splot_data = trait_data.xs("sPlot", axis=0, level=1)
            ts_data = trait_data.query(f"{trait_set_ids_col} == '{trait_set_id}'")
            y_positions.append(ts_data.pearsonr.values[-1])

            ax.plot(
                ts_data.resolution,
                ts_data.pearsonr,
                linestyle="-",
                color=color,
                label=f"{trait}",
                marker="o",
                markeredgecolor="white",
                markeredgewidth=0.7,
                ms=5,
                linewidth=0.75,
            )

            labels.append(trait_short)
            label_colors.append(color)

        texts = []
        for x_position, y_position, label, color in zip(
            x_positions, y_positions, labels, label_colors
        ):
            text = ax.text(
                x_position,
                y_position,
                label,
                ha="right",
                va="center",
                color=color,
                fontsize="x-small",
            )
            texts.append(text)

        # make sure the plots share the same y-axis
        if ncols > 1:
            ax.set_ylim(*Y_LIM)

        adjust_text_kwargs = {
            "force_text": (0, 0.5),
            "only_move": {"text": "y", "static": "y", "explode": "y", "pull": "y"},
        }

        adjust_text(
            texts,
            ax=ax,
            **adjust_text_kwargs,
        )

        # Readjust the x-position of the text since adjust_text doesn't seem to respect the
        # only_move parameter and still moves the text in the x-direction
        def _reset_text_x(_texts, _x_position, _ax):
            for _text in _texts:
                _text.set_ha("left")
                _text.set_x(_x_position)
                _text.set_transform(_ax.get_yaxis_transform())

        _reset_text_x(texts, text_x, ax)

        ax.set_xticks(range(len(resolutions)), resolution_labels)
        ax.set_xticklabels(resolution_labels)

        if unit == "km":
            ax.set_xlabel("Resolution")
        else:
            ax.set_xlabel("Resolution [$\degree$]")
        ax.set_ylabel("Pearson's $r$")
        ax.set_title(f"{trait_set_id}", x=0.7)
        # ax.invert_xaxis()

        # Remove the gridlines
        ax.grid(False)
        sns.despine(ax=ax)

    # add space between plots
    plt.subplots_adjust(wspace=0.5)

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    return axes

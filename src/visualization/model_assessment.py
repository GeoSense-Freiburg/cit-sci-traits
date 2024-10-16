import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

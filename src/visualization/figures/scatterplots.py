import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import font_manager
from matplotlib.axes import Axes

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import get_latest_run, get_trait_models_dir
from src.utils.plotting_utils import add_trait_name, add_trait_set_abbr
from src.utils.trait_utils import get_trait_number_from_id

CFG = get_config()


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot observed vs. predicted values for all traits"
    )
    parser.add_argument("--ts", default="COMB", help="Trait set to plot")
    parser.add_argument("--save", action="store_true", help="Save the plot to a file")
    parser.add_argument("--show", action="store_true", help="Show the plot")
    return parser.parse_args()


def main(args: argparse.Namespace = cli()) -> None:
    log.info("Preparing data...")
    data = prep_data(trait_set=args.ts)
    log.info("Plotting data...")
    plot(data)
    if args.save:
        log.info("Saving plot...")
        plt.savefig(
            f"results/figures/obs_vs_pred_{args.ts}.png", dpi=300, bbox_inches="tight"
        )
    if args.show:
        log.info("Displaying plot...")
        plt.show()


def prep_data(trait_set: str) -> pd.DataFrame:
    keep_traits = [f"X{t}" for t in CFG.datasets.Y.traits]  # noqa: F841
    return (
        pd.read_parquet("results/all_results.parquet")
        .query("resolution == '1km' and transform == 'power'")
        .assign(base_trait_id=lambda df: df.trait_id.str.split("_").str[0])
        .query("base_trait_id in @keep_traits")
        .drop(columns=["base_trait_id"])
        .pipe(add_trait_name)
        .pipe(add_trait_set_abbr)
        .query("trait_set_abbr == @trait_set")
    )


def plot(data: pd.DataFrame):
    font_path = "/usr/share/fonts/opentype/freefont/FreeSans.otf"
    font_bold_path = "/usr/share/fonts/opentype/freefont/FreeSansBold.otf"
    font_manager.fontManager.addfont(font_path)
    font_manager.fontManager.addfont(font_bold_path)
    plt.rcParams["font.family"] = "FreeSans"

    valid_trait_numbers = [
        get_trait_number_from_id(tid)
        for tid in data.sort_values(by="trait_name").trait_id.unique()
    ]

    N_COLS = 5
    N_ROWS = int(np.ceil(len(valid_trait_numbers) / N_COLS))

    with sns.plotting_context("paper", font_scale=1.4):
        fig, axs = plt.subplots(
            nrows=N_ROWS,
            ncols=N_COLS,
            figsize=(N_COLS * 5, N_ROWS * 5),
            dpi=300,
        )
        for trait_num, ax in zip(valid_trait_numbers, axs.flatten()):
            trait_id = f"X{trait_num}_mean"
            trait_results = data.query(f"trait_id == '{trait_id}'").iloc[0]
            trait_name = trait_results.trait_name

            model_dir = get_latest_run(get_trait_models_dir(trait_id)) / "splot_gbif"
            ovp = pd.read_parquet(model_dir / "cv_obs_vs_pred.parquet")

            observed = ovp.obs
            predicted = ovp.pred

            plot_observed_vs_predicted(
                ax,
                observed,
                predicted,
                trait_name,
                name_color="black",
                log_xf=False,
                density=True,
                stats={
                    "R²": trait_results.r2,
                    "Pearson's r": trait_results.pearsonr,
                    "nRMSE": trait_results.norm_root_mean_squared_error,
                },
                show_x_label=True,
                show_y_label=True,
            )

        # Remove empty axes
        for ax in axs.flatten()[len(valid_trait_numbers) :]:
            fig.delaxes(ax)

        # Add padding between plots in the y direction
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # plt.tight_layout()


def plot_observed_vs_predicted(
    ax: plt.Axes,
    observed: pd.Series,
    predicted: pd.Series,
    name: str,
    name_color: str = "black",
    log_xf: bool = False,
    density: bool = False,
    stats: dict = {},
    show_x_label: bool = True,
    show_y_label: bool = True,
) -> Axes:
    """Plot observed vs. predicted values."""
    p_low = min(predicted.min(), observed.min())
    p_high = max(predicted.max(), observed.max())

    cmap = sns.cubehelix_palette(start=0.5, rot=-0.75, reverse=True, as_cmap=True)  # type: ignore
    if density:
        sns.kdeplot(x=predicted, y=observed, ax=ax, cmap=cmap, fill=True, thresh=0.0075)
    else:
        sns.scatterplot(x=predicted, y=observed, ax=ax, s=1, alpha=0.01, edgecolor=None)

    # Fit a regression line for observed vs. predicted values, plot the regression
    # line so that it spans the entire plot, and print the correlation coefficient
    # Get m and b using scipy.stats.lingress
    m, b = scipy.stats.linregress(predicted, observed)[:2]
    reg_line = [m * p_low + b, m * p_high + b]

    if log_xf:
        ax.loglog(
            [p_low, p_high], [p_low, p_high], color="black", ls="-.", lw=0.5, alpha=0.5
        )
        ax.loglog([p_low, p_high], reg_line, color="red", lw=0.5)
    else:
        ax.plot(
            [p_low, p_high], [p_low, p_high], color="black", ls="-.", lw=0.5, alpha=0.5
        )
        ax.plot([p_low, p_high], reg_line, color="red", lw=0.5)

    # make sure lines are positioned on top of kdeplot
    ax.set_zorder(1)

    if stats:
        for i, (key, value) in enumerate(stats.items()):
            ax.text(
                0.05,
                0.95 - i * 0.1,
                f"{key} = {value:.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontdict={"fontsize": "large"},
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

    # include legend items for the reg_line and the 1-to-1 line
    ax.legend(
        [
            ax.get_lines()[0],
            ax.get_lines()[1],
        ],
        ["1-to-1", "Regression"],
        loc="lower right",
        frameon=True,
    )

    # set informative axes and title
    if show_x_label:
        ax.set_xlabel("Predicted")
    else:
        ax.set_xlabel("")
    if show_y_label:
        ax.set_ylabel("Observed")
    else:
        ax.set_ylabel("")
    ax.set_title(name, color=name_color, fontsize="large", fontweight="bold")

    # Set limits with a small buffer (5% of the range)
    buffer = (p_high - p_low) * 0.05
    ax.set_xlim(p_low - buffer, p_high + buffer)
    ax.set_ylim(p_low - buffer, p_high + buffer)
    sns.despine()

    return ax


if __name__ == "__main__":
    main()

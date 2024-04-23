from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from src.utils.raster_utils import open_raster


def plot_rasters(
    files: list[str] | list[Path],
    coarsen: int | None = None,
    show_latitude_density: bool = True,
    ncols: int = 2,
    context: str = "notebook",
    font_scale: float = 1,
) -> None:
    """Plot a raster with a colorbar for each file in a list of files."""
    nrows = (len(files) - 1) // ncols + 1

    # Set plotting context
    sns.set_context(
        context, font_scale=font_scale  # pyright: ignore[reportArgumentType]
    )

    fig = plt.figure(figsize=(15 * ncols, 10 * nrows), dpi=200)
    outer_gs = gridspec.GridSpec(nrows, ncols, figure=fig)

    for i, file in enumerate(files):
        rast = open_raster(file, mask_and_scale=True)

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=outer_gs[i],
            height_ratios=[1, 0.05],
            width_ratios=[1, 0.125],
        )

        ax0 = fig.add_subplot(inner_gs[0, 0])
        ax1 = fig.add_subplot(inner_gs[0, 1])
        cax = fig.add_subplot(inner_gs[1, 0])

        if coarsen:
            rast = rast.coarsen(x=coarsen, y=coarsen, boundary="trim").mean()

        im = rast.plot(ax=ax0, robust=True, cmap="viridis", add_colorbar=False)
        ax0.set_title(rast.attrs["long_name"], fontsize="xx-large")

        if show_latitude_density:
            rast.mean("x").plot(ax=ax1, y="y", color="black")
            ax1.set_ylabel("")
            ax1.yaxis.tick_right()
            ax1.set_xlabel("Mean Value")
            ax1.margins(y=0)
            ax1.set_facecolor("#f0f0f0")
            ax1.set_title("")
        else:
            ax1.axis("off")

        # Add latitude lines at multiples of 20
        for lat in range(-80, 81, 20):
            ax0.axhline(lat, color="gray", linewidth=0.5)
            ax1.axhline(lat, color="gray", linewidth=0.5)

        fig.colorbar(im, cax=cax, orientation="horizontal")

    plt.tight_layout()
    plt.show()


# Plot raster data as histograms
def plot_raster_distributions(
    files: List[Path], coarsen: int = 1, ncols: int = 5
) -> None:
    """Plot the distribution of raster data for each file in a list of files."""
    nrows = (len(files) - 1) // ncols + 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(15 * ncols, 10 * nrows))
    axs = axs.flatten()

    for ax, file in zip(axs, files):
        rast = (
            open_raster(file, mask_and_scale=True)
            .coarsen(x=coarsen, y=coarsen, boundary="trim")
            .mean()
        )
        data = rast.to_dataframe(name=rast.attrs["long_name"]).dropna().values.flatten()
        sns.histplot(data, ax=ax, kde=True)
        ax.set_title(file.name)

    # Remove unused subplots
    for ax in axs[len(files) :]:
        ax.remove()

    plt.tight_layout()
    plt.show()

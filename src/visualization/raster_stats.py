import xarray as xr
from matplotlib import pyplot as plt


def plot_raster(
    rast: xr.DataArray | xr.Dataset,
    coarsen: int | None = None,
    show_latitude_density: bool = True,
) -> None:
    """Plot a raster with a colorbar."""
    _, axs = plt.subplots(1, 2, figsize=(10, 10))
    if coarsen:
        rast = rast.coarsen(x=coarsen, y=coarsen, boundary="trim").mean()

    rast.plot(ax=axs[0], robust=True, cbar_kwargs=dict(orientation="horizontal"))

    if show_latitude_density:
        # add a line plot indicating the average value of each row (latitude) along
        # the right y-axis
        mean_x_values = rast.mean(dim="x")
        axs[1].plot(
            mean_x_values.squeeze(), mean_x_values.y, color="black", linestyle="--"
        )
        axs[1].invert_yaxis()

    plt.show()

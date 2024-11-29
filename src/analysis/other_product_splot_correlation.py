from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import get_trait_map_fns, get_trait_maps_dir
from src.utils.raster_utils import open_raster
from src.utils.spatial_utils import lat_weights, weighted_pearson_r


def raster_correlation(
    fn_left: Path, fn_right: Path, resolution: int | float
) -> tuple[str, float]:
    """Calculate the weighted Pearson correlation coefficient between a pair of trait maps."""
    log.info("Loading and filtering data for %s...", fn_right.stem)
    r_left = open_raster(fn_left).sel(band=1)
    r_right = open_raster(fn_right).sel(band=1)

    # Ensure the rasters are aligned
    r_right = r_right.rio.reproject_match(r_left)

    df_left = (
        r_left.to_dataframe(name=f"left_{fn_left.stem}")
        .drop(columns=["band", "spatial_ref"])
        .dropna()
    )
    df_right = (
        r_right.to_dataframe(name=f"right_{fn_right.stem}")
        .drop(columns=["band", "spatial_ref"])
        .dropna()
    )

    log.info("Joining dataframes (%s)...", fn_right.stem)
    df = df_left.join(df_right, how="inner")

    lat_unique = df.index.get_level_values("y").unique()

    log.info("Calculating weights (%s)...", fn_right.stem)
    weights = lat_weights(lat_unique, resolution)

    log.info(
        "Calculating weighted Pearson correlation coefficient (%s)...", fn_right.stem
    )
    r = weighted_pearson_r(df, weights)

    log.info("Weighted Pearson correlation coefficient: %s", r)

    return fn_right.stem, r


def all_products_paths() -> list[Path]:
    """Get the paths to all products."""
    products_dir = Path("data/interim/other_trait_maps")
    data = []
    for subdir in products_dir.iterdir():
        if subdir.is_dir():
            for file in subdir.glob("**/*"):
                if file.is_file():
                    data.append(file)
    return data


def gather_results(target_res: int | float) -> pd.DataFrame:
    """Gather the results of the raster correlation analysis into a DataFrame."""
    splot_corr_path = Path("results/product_comparison.csv")
    if splot_corr_path.exists():
        log.info("Loading existing results...")
        splot_corr = pd.read_csv(splot_corr_path)
    else:
        splot_corr = pd.DataFrame(columns=["trait_id", "author", "r", "resolution"])

    for fn in all_products_paths():
        res = fn.parent.stem
        if res != str(target_res).replace(".", ""):
            continue
        trait_id, author = fn.stem.split("_")
        splot_path = get_trait_maps_dir("splot") / f"{trait_id}.tif"
        _, r = raster_correlation(splot_path, fn, target_res)

        row = {"trait_id": trait_id, "author": author, "r": r, "resolution": res}
        splot_corr = pd.concat([splot_corr, pd.DataFrame([row])])

    return splot_corr.astype(
        {"trait_id": str, "author": str, "r": np.float64, "resolution": str}
    ).drop_duplicates(ignore_index=True)


def main() -> None:
    cfg = get_config()
    results = gather_results(cfg.target_resolution)
    results.drop_duplicates().to_csv("results/product_comparison.csv", index=False)


if __name__ == "__main__":
    main()

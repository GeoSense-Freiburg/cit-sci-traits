from pathlib import Path

import numpy as np
import pandas as pd

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import get_trait_maps_dir
from src.utils.raster_utils import open_raster
from src.utils.stat_utils import power_transform
from src.utils.trait_utils import get_trait_number_from_id

cfg = get_config()


def raster_correlation(
    splot_fn: Path, product_fn: Path, trait_num: str
) -> tuple[str, float]:
    """Calculate the weighted Pearson correlation coefficient between a pair of trait maps."""
    log.info("Loading and filtering data for %s...", product_fn.stem)
    splot_r = open_raster(splot_fn).sel(band=1)
    product_band = 2 if "wolf" in product_fn.stem else 1
    product_r = open_raster(product_fn).sel(band=product_band)

    # Ensure the rasters are aligned
    # r_right = r_right.rio.reproject_match(r_left)

    splot_df = (
        splot_r.to_dataframe(name=f"left_{splot_fn.stem}")
        .drop(columns=["band", "spatial_ref"])
        .dropna()
    )
    product_df_col_name = f"right_{product_fn.stem}"
    product_df = (
        product_r.to_dataframe(name=product_df_col_name)
        .drop(columns=["band", "spatial_ref"])
        .dropna()
    )

    # Transform the data if a transformer was used on the original sPlot data
    if cfg.trydb.interim.transform == "power":
        log.info("Power-transforming product data...")
        product_df[product_df_col_name] = power_transform(
            product_df[product_df_col_name].to_numpy(), trait_num
        )

    log.info("Joining dataframes (%s)...", product_fn.stem)
    df = splot_df.join(product_df, how="inner")

    log.info("Calculating Pearson correlation coefficient (%s)...", product_fn.stem)
    r = df.corr(method="pearson").iloc[0, 1]

    log.info("Pearson correlation coefficient: %s", r)

    return product_fn.stem, r


def all_products_paths() -> list[Path]:
    """Get the paths to all products."""
    products_dir = Path("data/interim/other_trait_maps")
    filepaths = []
    for subdir in products_dir.iterdir():
        if subdir.is_dir():
            for file in subdir.glob("**/*"):
                if file.is_file() and file.suffix == ".tif":
                    filepaths.append(file)
    return filepaths


def gather_results(target_res: str) -> pd.DataFrame:
    """Gather the results of the raster correlation analysis into a DataFrame."""
    splot_corr_path = Path("results/product_comparison.csv")
    dtypes = {"trait_id": str, "author": str, "r": np.float64, "resolution": str}
    if splot_corr_path.exists():
        log.info("Loading existing results...")
        splot_corr = pd.read_csv(splot_corr_path, dtype=dtypes)
    else:
        splot_corr = pd.DataFrame(
            columns=["trait_id", "author", "r", "resolution"], dtype=dtypes
        )

    for fn in all_products_paths():
        res = fn.parent.stem
        if res != str(target_res):
            continue
        trait_id, author = fn.stem.split("_")
        splot_path = get_trait_maps_dir("splot") / f"{trait_id}.tif"
        _, r = raster_correlation(splot_path, fn, get_trait_number_from_id(trait_id))

        row = {"trait_id": trait_id, "author": author, "r": r, "resolution": res}
        splot_corr = pd.concat([splot_corr, pd.DataFrame([row])])

    return splot_corr.astype(
        {"trait_id": str, "author": str, "r": np.float64, "resolution": str}
    ).drop_duplicates(ignore_index=True)


def main() -> None:
    cfg = get_config()
    results = gather_results(cfg.model_res)
    results.drop_duplicates().to_csv("results/product_comparison.csv", index=False)


if __name__ == "__main__":
    main()

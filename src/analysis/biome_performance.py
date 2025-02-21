import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from dask import compute, delayed

from src.conf.conf import get_config
from src.conf.environment import log
from src.models.cv_stats import get_stats
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import (
    get_all_aoa,
    get_all_cov,
    get_all_trait_models,
    get_biome_map_fn,
    get_latest_run,
    get_trait_models_dir,
)
from src.utils.raster_utils import open_raster

cfg = get_config()
N_FILES = None


# def cli() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Calculate model performance metrics per biome."
#     )
#     parser.add_argument("-o", "--overwrite", action="store_true")
#     return parser.parse_args()


def main() -> None:
    log.info("Calculating model stats per biome...")
    calc_model_stats_per_biome(overwrite=True)
    log.info("Calculating mean COV per biome...")
    calc_cov_per_biome()
    log.info("Calculating fraction inside AOA per biome...")
    calc_inside_aoa_pct_per_biome()


def calc_model_stats_per_biome(overwrite: bool = False) -> None:
    client, cluster = init_dask(dashboard_address=cfg.dask_dashboard)

    all_model_dirs = get_all_trait_models()
    if N_FILES is not None:
        all_model_dirs = list(all_model_dirs)[:N_FILES]

    tasks = [delayed(process_model_dir)(d) for d in all_model_dirs]
    results = compute(*tasks)

    close_dask(client, cluster)

    model_stats_df = pd.concat(results)
    all_biome_results_df = create_or_open_all_biome_results(overwrite=overwrite)

    all_biome_results_df = (
        pd.concat([all_biome_results_df, model_stats_df])
        .sort_values(by="run_id", ascending=False)
        .drop_duplicates(
            subset=["trait_id", "trait_set", "resolution", "pft", "transform", "biome"]
        )
    )
    results_path = Path("results/all_biome_results.parquet")
    all_biome_results_df.to_parquet(results_path)


def calc_cov_per_biome() -> None:
    all_cov_maps = get_all_cov()
    if N_FILES is not None:
        all_cov_maps = list(all_cov_maps)[:N_FILES]

    merge_cols = [
        "biome",
        "trait_id",
        "trait_set",
        "run_id",
        "pft",
        "resolution",
        "transform",
    ]
    client, cluster = init_dask(
        dashboard_address=cfg.dask_dashboard, n_workers=8, threads_per_worker=1
    )
    tasks = [delayed(process_cov)(raster) for raster in all_cov_maps]
    results = compute(*tasks)
    close_dask(client, cluster)
    all_cov_means = pd.concat(results)
    all_biome_results_df = create_or_open_all_biome_results()
    all_biome_results_df = pd.merge(
        all_biome_results_df, all_cov_means, on=merge_cols, how="left"
    )
    results_path = Path("results/all_biome_results.parquet")
    all_biome_results_df.to_parquet(results_path)


def calc_inside_aoa_pct_per_biome() -> None:
    all_aoa_maps = get_all_aoa()
    if N_FILES is not None:
        all_aoa_maps = list(all_aoa_maps)[:N_FILES]

    merge_cols = [
        "biome",
        "trait_id",
        "trait_set",
        "run_id",
        "pft",
        "resolution",
        "transform",
    ]
    client, cluster = init_dask(
        dashboard_address=cfg.dask_dashboard, n_workers=8, threads_per_worker=1
    )
    tasks = [delayed(process_aoa)(raster) for raster in all_aoa_maps]
    results = compute(*tasks)
    close_dask(client, cluster)
    all_aoa_fracs = pd.concat(results)
    all_biome_results_df = create_or_open_all_biome_results()
    all_biome_results_df = pd.merge(
        all_biome_results_df, all_aoa_fracs, on=merge_cols, how="left"
    )
    results_path = Path("results/all_biome_results.parquet")
    all_biome_results_df.to_parquet(results_path)


def create_or_open_all_biome_results(overwrite: bool = False) -> pd.DataFrame:
    if overwrite:
        return pd.DataFrame()
    results_path = Path("results/all_biome_results.parquet")
    if results_path.exists():
        return pd.read_parquet(results_path)
    else:
        return pd.DataFrame()


def process_model_dir(model_dir):
    trait_set = model_dir.name
    trait_id = model_dir.parents[2].name
    run_id = model_dir.parent.name
    pft = model_dir.parents[4].name
    ovp = (
        pd.read_parquet(model_dir / "cv_obs_vs_pred.parquet")
        .pipe(assign_biome_to_points)
        .query("biome != 98 and biome != 0")
    )
    biome_stats_df = pd.DataFrame()
    for biome in ovp.biome.unique():
        biome_df = ovp.query(f"biome == {biome}")
        stats = get_stats(biome_df)
        stats["biome"] = biome
        # stats is a dict in the form of {"biome": <biome>, "r2": <r2>, "pearsonr": <pearsonr>}
        # append to a dataframe where each row corresponds to a dict and the columns to the keys
        biome_stats_df = pd.concat([biome_stats_df, pd.DataFrame(stats, index=[0])])

    return biome_stats_df.assign(
        trait_id=trait_id,
        trait_set=trait_set,
        run_id=run_id,
        resolution=cfg.model_res,
        pft=pft,
        transform=cfg.trydb.interim.transform,
    )


def assign_biome_to_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(by=["x", "y"])
    with rasterio.open(get_biome_map_fn()) as src:
        # Read the entire band once
        biome_band = src.read(1)
        # Compute row,col indices vectorized
        row_cols = [src.index(x, y) for x, y in zip(df["x"], df["y"])]
        # Assign biome value for each row,col
        df["biome"] = [biome_band[r, c] for r, c in row_cols]
    return df


def compute_means_per_discrete_class(discrete_array, continuous_array):
    """
    Returns a dictionary mapping each discrete value in `discrete_array`
    to the mean of `continuous_array` values in that class.
    """

    # Ensure inputs are flattened to 1D
    labels = discrete_array.ravel()
    values = continuous_array.ravel()

    combined = (
        pd.DataFrame({"biome": labels, "cov": values})
        .dropna()
        .astype({"biome": np.int8})
    )

    # # 1) Drop NaNs from the continuous array
    # valid_mask = ~np.isnan(values)
    # labels = labels[valid_mask]
    # values = values[valid_mask]

    # # 2) Drop NaNs from the discrete array
    # valid_mask = ~np.isnan(labels)
    # labels = labels[valid_mask]
    # values = values[valid_mask]

    labels = combined.biome.to_numpy()
    values = combined["cov"].to_numpy()

    # Compute sums and counts per label using np.bincount
    sums = np.bincount(labels, weights=values)
    counts = np.bincount(labels)

    # Calculate means, skipping any labels that don't appear
    biomes = []
    means = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        # Avoid division by zero
        biomes.append(label)
        if counts[label] > 0:
            means.append(sums[label] / counts[label])
        else:
            means.append(np.nan)

    return pd.DataFrame({"biome": biomes, "mean_cov": means})


def process_cov(fn: Path) -> pd.DataFrame:
    trait_set = fn.parent.name
    trait_id = fn.parents[1].name
    run_id = get_latest_run(get_trait_models_dir(trait_id)).name
    pft = fn.parents[4].name
    resolution = cfg.model_res
    transform = cfg.trydb.interim.transform

    biomes = open_raster(get_biome_map_fn())
    cov = open_raster(fn)
    biomes = biomes.rio.reproject_match(cov)
    df = (
        compute_means_per_discrete_class(
            biomes.sel(band=1).values, cov.sel(band=1).values
        )
        .query("biome != 98")
        .assign(
            trait_id=trait_id,
            trait_set=trait_set,
            run_id=run_id,
            pft=pft,
            resolution=resolution,
            transform=transform,
        )
    )

    biomes.close()
    cov.close()
    gc.collect()

    return df


def compute_aoa_frac_per_discrete_class(
    discrete_array: np.ndarray, aoa_array: np.ndarray
):
    # Ensure inputs are flattened to 1D
    biomes = discrete_array.ravel()
    aoa = aoa_array.ravel()

    combined = (
        pd.DataFrame({"biome": biomes, "aoa": aoa})
        .dropna()
        .astype({"biome": np.int8, "aoa": np.int8})
    )

    # AOA can be either 0 or 1. We want to know the fraction of 0s per biome and
    # the fraction of 0s per biome compared to the total number of observations
    # The dataframe should ultimately have the columns: biome, frac_within_biome,
    # frac_of_whole
    fracs = (
        combined.groupby("biome")
        .agg(
            aoa_frac=("aoa", lambda x: 1 - x.mean()),
        )
        .reset_index()
    )

    return fracs


def process_aoa(fn: Path) -> pd.DataFrame:
    trait_set = fn.parent.name
    trait_id = fn.parents[1].name
    run_id = get_latest_run(get_trait_models_dir(trait_id)).name
    pft = fn.parents[4].name
    resolution = cfg.model_res
    transform = cfg.trydb.interim.transform

    biomes = open_raster(get_biome_map_fn())
    aoa = open_raster(fn)
    biomes = biomes.rio.reproject_match(aoa)
    df = (
        compute_aoa_frac_per_discrete_class(
            biomes.sel(band=1).to_numpy(), aoa.sel(band=2).to_numpy()
        )
        .query("biome != 98")
        .assign(
            trait_id=trait_id,
            trait_set=trait_set,
            run_id=run_id,
            pft=pft,
            resolution=resolution,
            transform=transform,
        )
    )

    biomes.close()
    aoa.close()
    gc.collect()

    return df


if __name__ == "__main__":
    main()

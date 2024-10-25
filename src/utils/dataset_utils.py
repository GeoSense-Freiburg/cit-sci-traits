"""Get the filenames of datasets based on the specified stage of processing."""

from pathlib import Path
from typing import Generator

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from autogluon.tabular import TabularPredictor
from box import ConfigBox
from dask import compute, delayed
from tqdm import trange

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.raster_utils import open_raster

cfg = get_config()


def get_eo_fns_dict(
    stage: str, datasets: str | list[str] | None = None
) -> dict[str, list[Path]]:
    """
    Get the filenames of EO datasets for a given stage.
    """
    if isinstance(datasets, str):
        datasets = [datasets]

    stage_map = {
        "raw": {"path": Path(cfg.raw_dir), "ext": ".tif"},
        "interim": {
            "path": Path(cfg.interim_dir) / cfg.eo_data.interim.dir / cfg.model_res,
            "ext": ".tif",
        },
    }

    if stage not in stage_map:
        raise ValueError("Invalid stage. Must be one of 'raw', 'interim'.")

    fns = {}
    match stage:
        case "raw":
            for k, v in cfg.datasets.X.items():
                fns[k] = list(
                    stage_map[stage]["path"].glob(f"{v}/*{stage_map[stage]['ext']}")
                )

        case "interim":
            for k in cfg.datasets.X.keys():
                fns[k] = list(
                    stage_map[stage]["path"].glob(f"{k}/*{stage_map[stage]['ext']}")
                )

    if datasets is not None:
        fns = {k: v for k, v in fns.items() if k in datasets}

    return fns


def get_eo_fns_list(stage: str, datasets: str | list[str] | None = None) -> list[Path]:
    """
    Get the filenames of EO datasets for a given stage, flattened into a list.
    """
    fns = get_eo_fns_dict(stage, datasets)

    # Return flattened list of filenames
    return [fn for ds_fns in fns.values() for fn in ds_fns]


def map_da_dtype(fn: Path, band: int = 1, nchunks: int = 9) -> tuple[str, str]:
    """
    Get the data type map for a given file.

    Args:
        fn (Path): The file path.
        band (int): The band number.
        nchunks (int): The number of chunks.

    Returns:
        tuple[str, str]: A tuple containing the long name and data type as strings.
    """
    res = get_res(fn)

    data = open_raster(
        fn,
        chunks={"x": (360 / res) // nchunks, "y": (180 / res) // nchunks},
        mask_and_scale=False,
    )
    long_name: str = data.attrs["long_name"]

    if fn.stem[0] == "X":
        long_name = f"{fn.stem}_{long_name[band - 1]}"
        data.attrs["long_name"] = long_name
    else:
        band = 1  # Only traits have multiple bands

    dtype = str(data.sel(band=band).dtype)

    data.close()

    return long_name, dtype


def map_da_dtypes(
    fns: list[Path], band: int = 1, nchunks: int = 9, dask: bool = False
) -> dict[str, str]:
    """
    Map the data types of a list of files.

    Args:
        fns (list[Path]): A list of file paths.
        nchunks (int): The number of chunks.

    Returns:
        dict[str, str]: A dictionary mapping the long names to the data types.
    """
    if dask:
        dtypes = [delayed(map_da_dtype)(fn, band=band, nchunks=nchunks) for fn in fns]
        return dict(set(compute(*dtypes)))

    dtype_map: dict[str, str] = {}
    for fn in fns:
        long_name, dtype = map_da_dtype(fn, band=band, nchunks=nchunks)
        dtype_map[long_name] = dtype

    return dtype_map


def get_res(fn: Path) -> int | float:
    """
    Get the resolution of a raster.
    """
    data = open_raster(fn).sel(band=1)
    res = abs(data.rio.resolution()[0])
    data.close()
    del data
    return res


@delayed
def load_x_or_y_raster(
    fn: Path, band: int = 1, nchunks: int = 9
) -> tuple[str, xr.DataArray]:
    """
    Load a raster dataset using delayed computation.

    Parameters:
        fn (Path): Path to the raster dataset file.
        nchunks (int): Number of chunks to divide the dataset into for parallel processing.

    Returns:
        tuple[xr.DataArray, str]: A tuple containing the loaded raster data as a DataArray
            and the long_name attribute of the dataset.

    Raises:
        ValueError: If multiple files are found while opening the raster dataset.
    """
    res = get_res(fn)
    da = open_raster(
        fn,
        chunks={"x": (360 / res) // nchunks, "y": (180 / res) // nchunks},
        mask_and_scale=True,
    ).sel(band=band)

    long_name = da.attrs["long_name"]

    # If the file is a trait map, append the band stat to the dataarray name
    if fn.stem.startswith("X"):
        bands = da.attrs["long_name"]
        long_name = f"{fn.stem}_{bands[band - 1]}"
        da.attrs["long_name"] = long_name

    return long_name, xr.DataArray(da)


def get_dataset_idx(fn_group: list[Path]) -> tuple[int, int]:
    """Get the array position of the sPlot and GBIF trait maps in a pair of trait maps."""
    gbif_idx = [i for i, fn in enumerate(fn_group) if "gbif" in str(fn)][0]
    splot_idx = 1 - gbif_idx

    return splot_idx, gbif_idx


def merge_splot_gbif(
    splot_id: int, gbif_id: int, dax: list[xr.DataArray]
) -> xr.DataArray:
    """Merge sPlot and GBIF trait maps in favor of sPlot."""
    return xr.where(
        dax[splot_id].notnull(), dax[splot_id], dax[gbif_id], keep_attrs=True
    )


def merge_splot_gbif_sources(
    splot_id: int, gbif_id: int, dax: list[xr.DataArray]
) -> xr.DataArray:
    """Merge sPlot and GBIF source maps in favor of sPlot."""
    return xr.where(
        dax[splot_id].notnull(), "s", xr.where(dax[gbif_id].notnull(), "g", None)
    )


def load_rasters_parallel(
    fns: list[Path] | list[list[Path]],
    band: int = 1,
    nchunks: int = 9,
) -> xr.Dataset:
    """
    Load multiple raster datasets in parallel using delayed computation.

    Parameters:
        fns (list[Path]): List of paths to the raster dataset files.
        nchunks (int): Number of chunks to divide each dataset into for parallel processing.

    Returns:
        dict[str, xr.DataArray]: A dictionary where keys are the long_name attributes of
            the datasets and values are the loaded raster data as DataArrays.
    """
    das: dict[str, xr.DataArray] = dict(
        compute(*[load_x_or_y_raster(fn, band=band, nchunks=nchunks) for fn in fns])
    )

    return xr.Dataset(das)


def compute_partitions(ddf: dd.DataFrame) -> pd.DataFrame:
    """
    Compute the partitions of a Dask DataFrame and return the result as a Pandas DataFrame.

    Parameters:
        ddf (dd.DataFrame): The input Dask DataFrame.

    Returns:
        pd.DataFrame: The concatenated Pandas DataFrame containing all partitions of the
            input Dask DataFrame.
    """
    npartitions = ddf.npartitions
    dfs = [
        ddf.get_partition(i).compute()
        for i in trange(npartitions, desc="Computing partitions")
    ]
    return pd.concat(dfs)


def check_y_set(y_set: str) -> None:
    """Check if the specified y_set is valid."""
    y_sets = ["gbif", "splot", "splot_gbif"]
    if y_set not in y_sets:
        raise ValueError(f"Invalid y_set. Must be one of {y_sets}.")


def get_models_dir(config: ConfigBox = cfg) -> Path:
    """Get the path to the models directory for a specific configuration."""
    return Path(config.models.dir) / config.PFT / config.model_res


def get_trait_models_dir(trait: str, config: ConfigBox = cfg) -> Path:
    """Get the path to the models directory for a specific trait and ML architecture."""
    return get_models_dir(config) / trait / config.train.arch


def get_all_trait_models(config: ConfigBox = cfg) -> Generator[Path, None, None]:
    """Get all trait models from each trait_set for a specific configuration."""
    for model_dir in get_models_dir().glob("X*"):
        for ts_dir in get_latest_run(model_dir / config.train.arch).iterdir():
            yield ts_dir


def get_latest_run(runs_path: Path) -> Path:
    """Get latest run from a specified trait models path."""
    sorted_runs = sorted(
        [run for run in Path(runs_path).glob("*") if "tmp" not in run.name],
        reverse=True,
    )
    if not sorted_runs:
        raise FileNotFoundError("No runs found.")
    return sorted_runs[0]


def get_predict_mask_fn(config: ConfigBox = cfg) -> Path:
    """Get the path to the predict features mask file for a specific configuration."""
    return (
        Path(config.train.dir)
        / config.eo_data.predict.dir
        / config.model_res
        / config.eo_data.predict.mask_fn
    )


def get_predict_imputed_fn(config: ConfigBox = cfg) -> Path:
    """Get the path to the imputed predict features file for a specific configuration."""
    return (
        Path(config.train.dir)
        / config.eo_data.predict.dir
        / config.model_res
        / config.eo_data.predict.imputed_fn
    )


def get_cv_models_dir(predictor: TabularPredictor) -> Path:
    """Get the path to the best base model for cross-validation analysis."""
    # Select the best base model (non-ensemble) to ensure fold-specific models exist
    best_base_model = (
        predictor.leaderboard(refit_full=False)
        .pipe(lambda df: df[df["stack_level"] == 1])
        .pipe(lambda df: df.loc[df["score_val"].idxmax()])
        .model
    )

    return Path(predictor.path, "models", str(best_base_model))


def get_train_dir(config: ConfigBox = cfg) -> Path:
    """Get the path to the train directory for a specific configuration."""
    return Path(config.train.dir) / config.PFT / config.model_res


def get_y_fn(config: ConfigBox = cfg) -> Path:
    """Get the path to the train file for a specific configuration."""
    return get_train_dir(config) / config.train.Y.fn


def get_autocorr_ranges_fn(config: ConfigBox = cfg) -> Path:
    """Get the path to the autocorrelation ranges file for a specific configuration."""
    return get_train_dir(config) / config.train.spatial_autocorr


def get_cv_splits_dir(config: ConfigBox = cfg) -> Path:
    """Get the path to the CV splits directory for a specific configuration."""
    return get_train_dir(config) / config.train.cv_splits.dir


def get_processed_dir(config: ConfigBox = cfg) -> Path:
    """Get the path to the processed directory for a specific configuration."""
    return Path(config.processed.dir) / config.PFT / config.model_res


def get_aoa_dir(config: ConfigBox = cfg) -> Path:
    """Get the path to aoa directory for a specific configuration."""
    return get_processed_dir(config) / config.aoa.dir


def get_cov_dir(config: ConfigBox = cfg) -> Path:
    """Get the path to the cov directory for a specific configuration."""
    return get_processed_dir(config) / config.cov.dir


def get_splot_corr_fn(config: ConfigBox = cfg) -> Path:
    """Get the path to the sPlot correlation file for a specific configuration."""
    return get_processed_dir(config) / config.processed.splot_corr


def get_weights_fn(config: ConfigBox = cfg) -> Path:
    """Get the path to the weights file."""
    return get_train_dir(config) / config.train.weights.fn


def get_trait_maps_dir(y_set: str, config: ConfigBox = cfg) -> Path:
    """Get the path to the trait maps directory for a specific dataset (e.g. GBIF or sPlot)."""
    check_y_set(y_set)

    return (
        Path(config.interim_dir)
        / config[y_set].interim.dir
        / config[y_set].interim.traits
        / config.PFT
        / config.model_res
    )


def get_trait_map_fns(y_set: str, config: ConfigBox = cfg) -> list[Path]:
    """Get the filenames of trait maps."""
    trait_maps_dir = get_trait_maps_dir(y_set, config)

    return sorted(list(trait_maps_dir.glob("*.tif")))


def get_model_performance(
    trait_id: str, trait_set: str, config: ConfigBox = cfg
) -> pd.DataFrame:
    """Get the model performance metrics for a specific trait and dataset."""
    fn = (
        get_latest_run(get_trait_models_dir(trait_id, config))
        / trait_set
        / cfg.train.eval_results
    )
    return pd.read_csv(fn)


def get_all_model_perf_fn(config: ConfigBox = cfg, debug: bool = False) -> Path:
    """Get the path to the model performance file for a specific configuration."""
    if debug:
        return Path(config.analysis.dir, "debug", config.analysis.multires_results_fn)
    return Path(config.analysis.dir, config.analysis.multires_results_fn)


def get_all_model_perf(config: ConfigBox = cfg, debug: bool = False) -> pd.DataFrame:
    """Load the model performance DataFrame for all traits."""
    try:
        return pd.read_parquet(get_all_model_perf_fn(config, debug=debug))
    except FileNotFoundError:
        log.warning("Results file not found, returning empty DataFrame.")
        return pd.DataFrame()


def get_all_fi_fn(config: ConfigBox = cfg, debug: bool = False) -> Path:
    """Get the path to the feature importance file for a specific configuration."""
    if debug:
        return Path(config.analysis.dir, "debug", config.analysis.multires_fi_fn)
    return Path(config.analysis.dir, config.analysis.multires_fi_fn)


def get_all_fi(config: ConfigBox = cfg, debug: bool = False) -> pd.DataFrame:
    """Load the feature importance DataFrame for all traits."""
    try:
        return pd.read_parquet(get_all_fi_fn(config, debug=debug))
    except FileNotFoundError:
        log.warning("Feature importance file not found, returning empty DataFrame.")
        return pd.DataFrame()


def get_feature_importance(
    trait_id: str, trait_set: str, config: ConfigBox = cfg
) -> pd.DataFrame:
    """Get the feature importance for a specific trait and dataset."""
    fn = (
        get_latest_run(get_trait_models_dir(trait_id, config))
        / trait_set
        / cfg.train.feature_importance
    )
    return pd.read_csv(fn, index_col=0, header=[0, 1])


def read_trait_map(
    trait_id: str, y_set: str, config: ConfigBox = cfg, band: int | None = None
) -> xr.DataArray | xr.Dataset:
    """Get the path to a specific trait map."""
    check_y_set(y_set)
    fn = get_trait_maps_dir(y_set, config) / f"{trait_id}.tif"

    if band is not None:
        return open_raster(fn).sel(band=band)
    return open_raster(fn)


def get_predict_dir(config: ConfigBox = cfg) -> Path:
    """Get the path to the predicted trait directory for a specific configuration."""
    return get_processed_dir(config) / config.predict.dir


def add_cv_splits_to_column(
    df: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """Add the CV splits to the DataFrame as a new column."""
    for i, (_, test_idx) in enumerate(splits):
        df.loc[test_idx, "cv_split"] = i
    return df


def get_final_fns(config: ConfigBox = cfg) -> Generator[Path, None, None]:
    """Get the filenames of the final trait maps."""
    return Path(get_processed_dir(config) / config.public.local_dir).glob("*.tif")


if __name__ == "__main__":
    print(get_eo_fns_dict("interim"))

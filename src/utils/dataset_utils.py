"""Get the filenames of datasets based on the specified stage of processing."""

from pathlib import Path
import pickle

import dask.dataframe as dd
import pandas as pd
import xarray as xr
from box import ConfigBox
from dask import compute, delayed
from tqdm import trange
from autogluon.tabular import TabularPredictor

from src.conf.conf import get_config
from src.utils.raster_utils import open_raster


def get_eo_fns_dict(
    stage: str, datasets: str | list[str] | None = None
) -> dict[str, list[Path]]:
    """
    Get the filenames of EO datasets for a given stage.
    """
    cfg: ConfigBox = get_config()

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


def get_trait_map_fns(stage: str) -> list[Path]:
    """Get the filenames of trait maps for a given stage."""
    cfg = get_config()

    stage_map = {
        "interim": {
            "path": Path(cfg.interim_dir),
            "ext": ".tif",
        },
    }
    if stage not in stage_map:
        raise ValueError(f"Invalid stage. Must be one of {stage_map.keys()}.")

    if stage == "interim":
        trait_map_fns = []
        y_datasets = cfg.datasets.Y.use.split("_")
        for dataset in y_datasets:
            trait_maps_dir = (
                Path(cfg.interim_dir)
                / cfg[dataset].interim.dir
                / cfg[dataset].interim.traits
                / cfg.PFT
                / cfg.model_res
            )
            trait_map_fns += list(trait_maps_dir.glob(f"*{stage_map[stage]['ext']}"))

    # Sort trait_map_fns by number in file name (eg. X1, X2, X3)
    return sorted(trait_map_fns, key=lambda x: int(x.stem.split("X")[-1]))


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
def load_x_raster(fn: Path, nchunks: int = 9) -> tuple[str, xr.DataArray]:
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
    ).sel(band=1)

    long_name = da.attrs["long_name"]

    return long_name, xr.DataArray(da)


def group_y_fns(fns: list[Path]) -> list[list[Path]]:
    """Group y rasters by trait. I.e. if both sPlot and GBIF files exist for a trait,
    group them."""
    unique_traits = sorted(
        {fn.stem.split("_")[0] for fn in fns},
        key=lambda x: int(x.split("X")[-1]),
    )
    fns_grouped = [[fn for fn in fns if trait == fn.stem] for trait in unique_traits]

    return fns_grouped


@delayed
def load_y_raster(
    fn_group: list[Path], band: int = 1, nchunks: int = 9
) -> tuple[str, xr.DataArray]:
    """Load and process y rasters. If both sPlot and GBIF are present, merge them in
    favor of sPlot."""
    # find all matching files in fns
    if len(fn_group) == 0:
        raise ValueError("No files found")

    das = []
    for raster_file in fn_group:
        da = open_raster(
            raster_file,
            chunks={"x": 36000 // nchunks, "y": 18000 // nchunks},
            mask_and_scale=True,
        )

        long_name = da.attrs["long_name"]
        long_name = f"{raster_file.stem}_{long_name[band - 1]}"
        da.attrs["long_name"] = long_name

        das.append(da.sel(band=band))

    if len(das) == 1:
        return long_name, das[0]

    # Find the array position of the fn in trait_fns that contains "gbif"
    gbif_idx = [i for i, fn in enumerate(fn_group) if "gbif" in str(fn)][0]
    splot_idx = 1 - gbif_idx

    merged = xr.where(
        das[splot_idx].notnull(), das[splot_idx], das[gbif_idx], keep_attrs=True
    )

    for da in das:
        da.close()

    return long_name, merged


def load_rasters_parallel(
    fns: list[Path] | list[list[Path]],
    band: int = 1,
    nchunks: int = 9,
    ml_set: str = "x",
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
    if ml_set not in ["x", "y"]:
        raise ValueError("Invalid ml_set. Must be one of 'x', 'y'.")

    if ml_set == "x":
        das: dict[str, xr.DataArray] = dict(
            compute(*[load_x_raster(fn, nchunks) for fn in fns])
        )

    if ml_set == "y":
        das: dict[str, xr.DataArray] = dict(
            compute(*[load_y_raster(fn_group, band, nchunks) for fn_group in fns])
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


def get_models_dir(cfg: ConfigBox) -> Path:
    """Get the path to the models directory for a specific configuration."""
    return (
        Path(cfg.models.dir)
        / cfg.PFT
        / cfg.model_res
        / cfg.datasets.Y.use
        / cfg.train.arch
    )


def get_predict_fn(cfg: ConfigBox) -> Path:
    """Get the path to the predict file for a specific configuration."""
    return (
        Path(cfg.train.dir)
        / cfg.eo_data.predict.dir
        / cfg.model_res
        / cfg.eo_data.predict.filename
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


def get_train_dir(cfg: ConfigBox) -> Path:
    """Get the path to the train directory for a specific configuration."""
    return Path(cfg.train.dir) / cfg.PFT / cfg.model_res / cfg.datasets.Y.use


def get_train_fn(cfg: ConfigBox) -> Path:
    """Get the path to the train file for a specific configuration."""
    return get_train_dir(cfg) / cfg.train.features


def get_cv_splits(cfg: ConfigBox, label: str):
    """Load the CV splits for a given label."""
    with open(get_train_dir(cfg) / cfg.train.cv_splits.dir / f"{label}.pkl", "rb") as f:
        return pickle.load(f)


def get_processed_dir(cfg: ConfigBox) -> Path:
    """Get the path to the processed directory for a specific configuration."""
    return Path(cfg.processed.dir) / cfg.PFT / cfg.model_res / cfg.datasets.Y.use

def get_splot_corr_fn(cfg: ConfigBox) -> Path:
    """Get the path to the sPlot correlation file for a specific configuration."""
    return get_processed_dir(cfg) / cfg.processed.splot_corr


def get_weights_fn(cfg: ConfigBox) -> Path:
    """Get the path to the weights file."""
    return get_train_dir(cfg) / cfg.train.weights.fn


def get_trait_maps_dir(cfg: ConfigBox, dataset: str) -> Path:
    """Get the path to the trait maps directory for a specific dataset (e.g. GBIF or sPlot)."""
    if dataset not in cfg.datasets.Y.keys():
        raise ValueError(
            f"Invalid dataset. Must be one of {cfg.datasets.Y.keys()[:2]}."
        )

    return (
        Path(cfg.interim_dir)
        / cfg[dataset].interim.dir
        / cfg[dataset].interim.traits
        / cfg.PFT
        / cfg.model_res
    )


def get_predict_dir(cfg: ConfigBox) -> Path:
    """Get the path to the predicted trait directory for a specific configuration."""
    return (
        Path(cfg.processed.dir)
        / cfg.PFT
        / cfg.model_res
        / cfg.datasets.Y.use
        / cfg.processed.predict_dir
    )


if __name__ == "__main__":
    print(get_eo_fns_dict("interim"))

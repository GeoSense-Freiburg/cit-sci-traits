"""Reproject EO data to a target resolution, mask out non-vegetation pixels, and save as
a DataFrame."""

import argparse
import gc
import os
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

from src.conf.conf import get_config
from src.conf.environment import log
from src.data.get_dataset_filenames import get_dataset_filenames
from src.data.mask import get_mask, mask_raster
from src.utils.raster_utils import (
    create_sample_raster,
    open_raster,
    scale_data,
    xr_to_raster,
)


def cli() -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Reproject EO data to a DataFrame.")
    parser.add_argument(
        "-n",
        "--n-workers",
        type=int,
        default=1,
        help="Number of workers. Use -1 for all CPUs.",
    )
    parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run.")
    args = parser.parse_args()

    if args.n_workers <= 0 and args.n_workers != -1:
        raise ValueError("Number of workers must be either -1 or greater than 0.")

    return args


def process_file(
    filename: str | os.PathLike,
    dataset: str,
    mask: xr.DataArray,
    out_dir: str | Path,
    target_raster: xr.DataArray,
    dry_run: bool = False,
):
    """
    Process a file by reprojecting and masking a raster, converting it to a GeoDataFrame,
    optimizing the data types of the columns, and writing it to a Parquet file.

    Args:
        filename (str or os.PathLike): The path to the input raster file.
        mask (xr.DataArray): The mask to apply to the raster.
        out_dir (str or Path): The directory where the output Parquet file will be saved.
        target_raster (xr.DataArray): The target raster to match the resolution of the
            masked raster.
        dry_run (bool, optional): If True, the function will only perform a dry run
            without writing the output file. Defaults to False.
    """
    filename = Path(filename)
    rast = open_raster(filename).sel(band=1).rio.reproject_match(mask)
    rast_masked = mask_raster(rast, mask)

    rast.close()
    mask.close()
    del rast, mask

    if rast_masked.rio.resolution() != target_raster.rio.resolution():
        rast_masked = rast_masked.rio.reproject_match(target_raster)

    dtype = rast_masked.dtype

    if dataset == "modis":
        # Values outside this range usually represent errors in the atmospheric correction
        # algorithm
        rast_masked = rast_masked.clip(0, 10000)
        dtype = "int16"

    if dataset == "soilgrids":
        dtype = "int16"
        # some soil properties have smaller ranges
        if (
            rast_masked.max() < np.iinfo(np.int8).max
            and rast_masked.min() >= np.iinfo(np.int8).min
        ):
            dtype = "int8"

    if dataset == "canopy_height":
        dtype = "uint8"

    if dataset == "vodca":
        dtype = "int16"
        rast_masked = scale_data(rast_masked, dtype, True)

    if "long_name" not in rast_masked.attrs:
        rast_masked.attrs["long_name"] = Path(filename).stem

    out_path = Path(out_dir) / dataset / f"{Path(filename).stem}.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        xr_to_raster(
            rast_masked, out_path, compression_level=18, num_threads=1, dtype=dtype
        )

    rast_masked.close()
    del rast_masked

    gc.collect()


def modis_ndvi(out_dir: Path, dry_run: bool = False) -> None:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI) from MODIS satellite data.

    Parameters:
    - out_dir (Path): The output directory where the NDVI raster will be saved.
    """
    for month in range(1, 13):
        fns = sorted(list(out_dir.glob(f"modis/*_m{month}_*.tif")))
        out_path = fns[0].parent / fns[0].name.replace("b01", "ndvi")
        fns = sorted(list((out_dir / "modis").glob(f"*_m{month}_*.tif")))

        red = open_raster(fns[0]).sel(band=1)
        nir = open_raster(fns[1]).sel(band=1)
        ndvi = (nir - red) / (nir + red)

        # Scale the values prior to int conversion
        ndvi = ndvi * 10000
        ndvi.attrs["long_name"] = out_path.stem

        if not dry_run:
            xr_to_raster(ndvi, out_path, dtype="int16")

        # Clean up
        for da in [red, nir, ndvi]:
            da.close()

        del red, nir, ndvi
        gc.collect()


def prune_worldclim(out_dir: Path, bio_vars: List[str], dry_run: bool = False) -> None:
    """
    Prunes WorldClim data files based on the specified bio_vars.

    Args:
        out_dir (Path): The output directory where the pruned files will be saved.
        bio_vars (List[str]): A list of bio_vars to be pruned.
    """
    fns = list(out_dir.glob("worldclim/*.tif"))

    for bio_var in bio_vars:
        if "-" in bio_var:
            start, end = bio_var.split("-")
            da1 = open_raster([fn for fn in fns if f"bio_{start}" in fn.name][0]).sel(
                band=1
            )
            da2 = open_raster([fn for fn in fns if f"bio_{end}" in fn.name][0]).sel(
                band=1
            )
            diff = da1 - da2
            diff_name = f"wc2.1_30s_bio_{start}-{end}"
            diff.attrs["long_name"] = diff_name

            if not dry_run:
                xr_to_raster(diff, out_dir / "worldclim" / f"{diff_name}.tif")

            for da in [da1, da2, diff]:
                da.close()

            del da1, da2, diff
            gc.collect()

    # Delete files that don't contain a bio_var
    for fn in fns:
        if not any(f"bio_{var}.tif" in fn.name for var in bio_vars) and not dry_run:
            fn.unlink()


def main(args: argparse.Namespace) -> None:
    """Main function."""
    cfg = get_config()

    log.info("Collecting files...")
    filenames = get_dataset_filenames(stage="raw")

    out_dir = Path(cfg.eo_data.interim.dir) / cfg.model_res

    if not filenames:
        log.error("No files to process.")
        return

    log.info("Building reference rasters...")
    target_sample_raster = create_sample_raster(resolution=cfg.target_resolution)

    log.info("Building landcover mask...")
    mask = get_mask(cfg.mask.path, cfg.mask.keep_classes, cfg.base_resolution)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Harmonizing rasters...")
    tasks = [
        delayed(process_file)(
            filename, dataset, mask, out_dir, target_sample_raster, args.dry_run
        )
        for dataset, ds_fns in filenames.items()
        for filename in ds_fns
    ]
    Parallel(n_jobs=args.n_workers)(tqdm(tasks, total=len(tasks)))

    if "modis" in cfg.datasets.X:
        log.info("Calculating MODIS NDVI...")
        modis_ndvi(out_dir)

    if "worldclim" in cfg.datasets.X:
        log.info("Calculating WorldClim bioclimatic variables...")
        prune_worldclim(out_dir, cfg.worldclim.bio_vars)

    log.info("Done. ✅")


if __name__ == "__main__":
    cli_args = cli()
    main(cli_args)

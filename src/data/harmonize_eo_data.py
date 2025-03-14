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
from src.conf.environment import detect_system, log
from src.data.mask import get_mask, mask_raster
from src.utils.dataset_utils import get_eo_fns_dict
from src.utils.raster_utils import (
    create_sample_raster,
    open_raster,
    pack_xr,
    xr_to_raster,
    xr_to_raster_rasterio,
)


def process_file(
    filename: str | os.PathLike,
    dataset: str,
    mask: xr.DataArray,
    out_dir: str | Path,
    target_raster: xr.DataArray,
    dry_run: bool = False,
    overwrite: bool = False,
):
    """
    Process a file by reprojecting and masking a raster.

    Args:
        filename (str or os.PathLike): The path to the input raster file.
        dataset (str): The dataset to which the raster belongs.
        mask (xr.DataArray): The mask to apply to the raster.
        out_dir (str or Path): The directory where the output Parquet file will be saved.
        target_raster (xr.DataArray): The target raster to match the resolution of the
            masked raster.
        dry_run (bool, optional): If True, the function will only perform a dry run
            without writing the output file. Defaults to False.
        overwrite (bool, optional): If True, the function will overwrite the output file.
    """
    filename = Path(filename)
    out_path = Path(out_dir) / dataset / f"{Path(filename).stem}.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        log.info("Skipping %s...", filename)
        return

    rast = open_raster(filename).sel(band=1)

    if rast.rio.nodata is None:
        # Make sure that the raster has a nodata value or else the reproject_match
        # method will treat nan values as actual data
        rast = rast.rio.write_nodata(np.nan)

    rast = rast.rio.reproject_match(mask)
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

    if dataset in ("worldclim", "vodca"):
        dtype = "int16"
        rast_masked = pack_xr(rast_masked)

    if "long_name" not in rast_masked.attrs:
        rast_masked.attrs["long_name"] = Path(filename).stem

    if not dry_run:
        match dtype:
            case "uint8":
                nodata = np.iinfo(np.uint8).max
            case "int8":
                nodata = np.iinfo(np.int8).min
            case "int16":
                nodata = np.iinfo(np.int16).min
            case _:
                nodata = np.nan
        # xr_to_raster(
        #     rast_masked,
        #     out_path,
        #     compression_level=18,  # pyright: ignore[reportArgumentType]
        #     num_threads=1,
        #     dtype=dtype,
        # )
        xr_to_raster_rasterio(rast_masked, out_path, nodata=nodata)

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


def cli() -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Reproject EO data to a DataFrame.")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run.")
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite files."
    )
    args = parser.parse_args()

    return args


def main(args: argparse.Namespace) -> None:
    """Main function."""
    cfg = get_config()
    syscfg = cfg[detect_system()][cfg.model_res].harmonize_eo_data

    if syscfg.n_workers == -1:
        syscfg.n_workers = os.cpu_count()

    log.info("Collecting files...")
    filenames = get_eo_fns_dict(stage="raw")

    out_dir = Path(cfg.interim_dir) / cfg.eo_data.interim.dir / cfg.model_res

    if not filenames:
        log.error("No files to process.")
        return

    log.info("Building reference rasters...")
    target_sample_raster = create_sample_raster(
        resolution=cfg.target_resolution, crs=cfg.crs
    )

    log.info("Building landcover mask...")
    mask = get_mask(cfg.mask.path, cfg.mask.keep_classes, cfg.base_resolution, cfg.crs)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Harmonizing rasters...")
    tasks = [
        delayed(process_file)(
            filename,
            dataset,
            mask,
            out_dir,
            target_sample_raster,
            args.dry_run,
            args.overwrite,
        )
        for dataset, ds_fns in filenames.items()
        for filename in ds_fns
    ]
    Parallel(n_jobs=syscfg.n_workers)(tqdm(tasks, total=len(tasks)))

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

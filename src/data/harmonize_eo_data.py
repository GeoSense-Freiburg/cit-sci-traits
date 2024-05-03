"""Reproject EO data to a target resolution, mask out non-vegetation pixels, and save as
a DataFrame."""

import argparse
import gc
import multiprocessing
import os
from pathlib import Path

import xarray as xr

from src.conf.conf import get_config
from src.conf.environment import log
from src.data.get_dataset_filenames import get_dataset_filenames
from src.data.mask import get_mask, mask_raster
from src.utils.df_utils import optimize_columns, write_df
from src.utils.log_utils import setup_logger
from src.utils.raster_utils import create_sample_raster, open_raster, raster_to_df


def cli() -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Reproject EO data to a DataFrame.")
    parser.add_argument(
        "-n", "--n-workers", type=int, default=1, help="Number of workers."
    )
    parser.add_argument(
        "-r", "--resume", action="store_true", help="Resume processing."
    )
    parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run.")
    args = parser.parse_args()

    if args.n_workers <= 0 and args.n_workers != -1:
        raise ValueError("Number of workers must be either -1 or greater than 0.")

    if args.n_workers == -1:
        args.n_workers = multiprocessing.cpu_count()

    return args


def process_file(
    filename: str | os.PathLike,
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
    if __name__ == "__main__":
        proc_log = log
    else:
        proc_log = setup_logger(__name__, "INFO")

    filename = Path(filename)
    proc_log.info("Processing %s...", filename)
    try:
        rast = open_raster(filename).sel(band=1).rio.reproject_match(mask)
        masked = mask_raster(rast, mask)

        rast.close()
        mask.close()
        del rast, mask

        if masked.rio.resolution() != target_raster.rio.resolution():
            proc_log.warning(
                "Reproject masked raster: %s -> %s",
                masked.rio.resolution(),
                target_raster.rio.resolution(),
            )
            masked = masked.rio.reproject_match(target_raster)

        if "long_name" not in masked.attrs:
            masked.attrs["long_name"] = Path(filename).stem

        proc_log.info("Converting %s to DataFrame...", filename.name)
        df = raster_to_df(masked)
        masked.close()
        del masked

        proc_log.info("Optimizing dtype for %s...", filename.name)
        df = optimize_columns(df, coords_as_categories=True)

        dataset_dir = Path(filename).parent.name
        dataset_dir = (
            dataset_dir.replace("_1km", "") if "_1km" in dataset_dir else dataset_dir
        )

        out_path = Path(out_dir) / dataset_dir / f"{Path(filename).stem}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        log.info("Writing %s...%s", out_path, " (dry run)" if dry_run else "")
        if not dry_run:
            write_df(df, out_path, writer="parquet", dask=False)

        del df
        gc.collect()

    except Exception as e:  # pylint: disable=broad-exception-caught
        proc_log.error("Error processing %s: %s", filename, e)


def main(args: argparse.Namespace) -> None:
    """Main function."""
    conf = get_config()

    log.info("Collecting files...")
    filenames = get_dataset_filenames(conf.datasets.X, stage="raw")

    out_dir = Path(conf.interim.eo_data) / conf.model_res

    if args.resume:
        processed_files = list(out_dir.rglob("*/*.parquet"))
        processed_files = [f.stem for f in processed_files]
        filenames = [f for f in filenames if f.stem not in processed_files]

    if not filenames:
        log.error("No files to process.")
        return

    log.info("Building reference rasters...")
    base_sample_raster = create_sample_raster(conf.extent, conf.base_resolution)
    target_sample_raster = create_sample_raster(conf.extent, conf.target_resolution)

    log.info("Generating landcover mask...")
    mask = get_mask(conf.mask.path, conf.mask.keep_classes, base_sample_raster)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.n_workers > 1:
        with multiprocessing.Pool(args.n_workers) as pool:
            pool.starmap(
                process_file,
                [
                    (filename, mask, out_dir, target_sample_raster, args.dry_run)
                    for filename in filenames
                ],
            )
    else:
        for filename in filenames:
            process_file(filename, mask, out_dir, target_sample_raster, args.dry_run)


if __name__ == "__main__":
    cli_args = cli()
    main(cli_args)

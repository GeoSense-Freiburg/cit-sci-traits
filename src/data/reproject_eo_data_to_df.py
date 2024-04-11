import argparse
import gc
import multiprocessing
import os

from src.conf.conf import get_config
from src.conf.environment import log
from src.data.get_dataset_filenames import get_dataset_filenames
from src.data.mask import mask_non_vegetation
from src.utils.log_utils import subprocess_logger
from src.utils.raster_utils import open_raster


def cli() -> argparse.Namespace:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Reproject EO data to a DataFrame.")
    parser.add_argument(
        "-n", "--n-workers", type=int, default=-1, help="Number of workers."
    )
    args = parser.parse_args()

    if args.n_workers <= 0 and args.n_workers != -1:
        raise ValueError("Number of workers must be either -1 or greater than 0.")

    if args.n_workers == -1:
        args.n_workers = multiprocessing.cpu_count()

    return args


def process_file(filename: str | os.PathLike, conf: dict):
    # Check if in subprocess
    proc_log = log

    if "subprocess" in __name__:
        proc_log = subprocess_logger(__name__)

    rast = open_raster(filename)
    masked = mask_non_vegetation(rast, conf.mask)
    reprojected = reproject_to_reference(masked, conf.resolution)
    df = raster_to_df(reprojected)
    write_df(df, engine="parquet")
    del rast, masked, reprojected, df
    gc.collect()
    proc_log.info("Processed %s", filename)


def main(args: argparse.Namespace) -> None:
    conf = get_config()

    log.info("Collecting files...")
    filenames = get_dataset_filenames(conf.datasets.X)

    for filename in filenames:
        log.info(filename)

        process_file(filename, conf)

    # reproject_and_write_as_df(filenames, conf.resolution)


if __name__ == "__main__":
    cli_args = cli()
    main(cli_args)

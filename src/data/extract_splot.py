"""Extracts and converts the sPlot data to pandas DataFrames."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from box import ConfigBox
from rpy2.robjects import pandas2ri

from src.conf.conf import get_config
from src.conf.environment import log


def main(cfg: ConfigBox = get_config()) -> None:
    """Extracts and converts the sPlot data to pandas DataFrames."""
    splot_raw_dir = Path(cfg.raw_dir, cfg.datasets.Y.splot)
    splot_prep_dir = (
        Path(cfg.interim_dir, cfg.splot.interim.dir) / cfg.splot.interim.extracted
    )
    splot_prep_dir.mkdir(parents=True, exist_ok=True)
    zip_path = splot_raw_dir / "extracted_data.zip"
    extracted_rdata_fp = splot_prep_dir / "extracted_data.RData"

    # 01. Extract sPlot data
    log.info("Unzipping %s to %s... ", zip_path, splot_prep_dir)
    # Use local `unzip` as `zipfile` cannot unzip LZMA-compressed data
    subprocess.run(
        ["unzip", "-o", str(zip_path), "-d", str(splot_prep_dir)], check=False
    )

    # 02. Load sPlot data with R, coerce datetimes to strings
    log.info("Loading unzipped sPlot data...")
    r_unzip_data = f"""
    splot_prep_dir <- "{str(splot_prep_dir)}"
    splot_rdata_fp <- file.path("{str(extracted_rdata_fp)}")
    load(splot_rdata_fp)
    header$Date <- as.character(header$Date)
    """
    ro.r(r_unzip_data)

    header = ro.r["header"]
    vegetation = ro.r["vegetation"]
    trait = ro.r["trait"]

    # 03. Convert R dataframes to pandas DataFrames
    log.info("Converting sPlot dataframes to pandas DataFrames...")
    with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
        header_df = ro.conversion.get_conversion().rpy2py(header)
        vegetation_df = ro.conversion.get_conversion().rpy2py(vegetation)
        trait_df = ro.conversion.get_conversion().rpy2py(trait)

    # 04. Optimize DataFrames
    log.info("Optimizing DataFrames...")
    for df in [header_df, vegetation_df, trait_df]:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string[pyarrow]")

                # Now replace "NA_character_" instances with np.nan
                df[col] = df[col].replace({"NA_character_": np.nan, "NA": np.nan})

            # and a little more data-savings
            if col == "PlotObservationID":
                df[col] = df[col].astype("int64").astype("string[pyarrow]")

            if col in ["Latitude", "Longitude"]:
                df[col] = df[col].astype("category")

            if col == "Date":
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")

    # 05. Save DataFrames to parquet
    log.info("Saving DataFrames to disk...")
    header_df.to_parquet(splot_prep_dir / "header.parquet", compression="zstd")
    vegetation_df.to_parquet(splot_prep_dir / "vegetation.parquet", compression="zstd")
    trait_df.to_parquet(splot_prep_dir / "trait.parquet", compression="zstd")

    # 06. Clean up
    extracted_rdata_fp.unlink()
    log.info("Done.")


if __name__ == "__main__":
    main()

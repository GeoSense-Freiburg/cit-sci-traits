"""Combines final data products (e.g. predicted traits and CoV), adds nice metadata,
and uploads to a target destination for sharing."""

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from box import ConfigBox
from dask import compute, delayed
from scipy.spatial.distance import cdist

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.io.upload_sftp import upload_file_sftp
from src.utils.dask_utils import close_dask, init_dask
from src.utils.dataset_utils import (
    get_model_performance,
    get_predict_dir,
    get_processed_dir,
)
from src.utils.raster_utils import pack_data
from src.utils.spatial_utils import interpolate_like
from src.utils.trait_utils import get_trait_number_from_id


@delayed
def process_single_trait_map(
    trait_map: Path,
    trait_set: str,
    cfg: ConfigBox,
    metadata: dict,
    trait_mapping: dict,
    trait_agg: dict,
    destination: str = "local",
    overwrite: bool = False,
) -> None:
    """Process a single trait map."""
    log.info("Processing %s", trait_map)

    if destination not in ["sftp", "local", "both"]:
        raise ValueError("Invalid destination. Must be one of 'sftp', 'local', 'both'.")

    output_file_name = f"{trait_map.stem}_{cfg.PFT}_{cfg.model_res}_deg.tif"

    if destination in ("local", "both") and not overwrite:
        local_dest_dir = get_processed_dir() / cfg.public.local_dir
        if (local_dest_dir / output_file_name).exists():
            log.info("File already exists. Skipping...")
            return

    trait_num = get_trait_number_from_id(trait_map.stem)
    trait_meta = trait_mapping[trait_num]
    metadata["trait_short_name"] = trait_meta["short"]
    metadata["trait_long_name"] = trait_meta["long"]
    metadata["trait_unit"] = trait_meta["unit"]

    ds_count = 0
    log.info("Reading and packing predict...")
    predict_path = trait_map / trait_set / f"{trait_map.stem}_{trait_set}_predict.tif"
    predict_ds = rasterio.open(predict_path, "r")
    predict = predict_ds.read(1)
    predict_valid_mask = ~np.isnan(predict)

    scales = []
    offsets = []
    predict = pack_data(predict_ds.read(1))
    scales.append(predict[0])
    offsets.append(predict[1])
    nodata = predict[2]
    predict = predict[3]

    ds_count += 1

    log.info("Reading and packing CoV...")
    cov_path = (
        trait_map.parents[1]
        / cfg.cov.dir
        / trait_map.stem
        / trait_set
        / f"{trait_map.stem}_{trait_set}_cov.tif"
    )
    cov_dataset = rasterio.open(cov_path, "r")
    cov = pack_data(cov_dataset.read(1))
    scales.append(cov[0])
    offsets.append(cov[1])
    cov = cov[3]
    ds_count += 1

    log.info("Reading AoA...")
    aoa_path = (
        trait_map.parents[1]
        / cfg.aoa.dir
        / trait_map.stem
        / trait_set
        / f"{trait_map.stem}_{trait_set}_aoa.tif"
    )
    aoa_dataset = rasterio.open(aoa_path, "r")
    log.info("Interpolating AoA...")
    aoa = interpolate_like(aoa_dataset.read(2), predict_valid_mask)

    log.info("Masking and downcasting AoA...")
    aoa = np.where(np.isnan(aoa), nodata, aoa).astype("int16")
    scales.append(1)
    offsets.append(0)
    ds_count += 1

    log.info("Gathering model performance metrics...")
    mp = get_model_performance(trait_map.stem, trait_set).query("transform == 'none'")
    mp_dict = {
        "R2": mp["r2"].values[0].round(2),
        "Pearson's r": mp["pearsonr_wt"].values[0].round(2),
        "nRMSE": mp["norm_root_mean_squared_error"].values[0].round(2),
        "RMSE": mp["root_mean_squared_error"].values[0].round(2),
    }

    log.info("Generating metadata...")
    bounds = predict_ds.bounds
    spatial_extent = (
        f"min_x: {bounds.left}, min_y: {bounds.bottom}, "
        f"max_x: {bounds.right}, max_y: {bounds.top}"
    )

    raster_meta = {
        "crs": predict_ds.crs,
        "resolution": predict_ds.res,
        "geospatial_units": "degrees",
        "grid_coordinate_system": "WGS 84",
        "transform": predict_ds.transform,
        "nodata": nodata,
        "spatial_extent": spatial_extent,
        "model_performance": json.dumps(mp_dict),
    }

    # Read data from the original file
    cog_profile = predict_ds.profile.copy()
    cog_profile.update(count=ds_count)

    # Ensure new profile is configured to write as a COG
    cog_profile.update(
        driver="GTiff",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="ZSTD",
        copy_src_overviews=True,
        interleave="band",
        nodata=nodata,
    )

    new_tags = predict_ds.tags().copy()

    for key, value in metadata.items():
        new_tags[key] = value

    for key, value in raster_meta.items():
        new_tags[key] = value

    log.info("Writing new file...")
    with tempfile.TemporaryDirectory() as temp_dir:
        new_file_path = Path(
            temp_dir,
            f"{trait_map.stem}_{cfg.PFT}_{cfg.model_res}_deg.tif",
        )

        # Create a new file with updated metadata and original data
        with rasterio.open(
            new_file_path,
            "w",
            **cog_profile,
        ) as new_dataset:
            new_dataset.update_tags(**new_tags)

            log.info("Setting scales and offsets...")
            scales = np.array(scales, dtype=np.float64)
            offsets = np.array(offsets, dtype=np.float64)
            new_dataset._set_all_scales(scales)  # pylint: disable=protected-access
            new_dataset._set_all_offsets(offsets)  # pylint: disable=protected-access

            for i in range(1, ds_count + 1):
                new_dataset.update_tags(i, _FillValue=nodata)

            log.info("Writing bands...")
            new_dataset.write(predict, 1)
            new_dataset.write(cov, 2)
            new_dataset.write(aoa, 3)

            new_dataset.set_band_description(
                1,
                f"{trait_meta['short']} ({trait_agg[str(cfg.datasets.Y.trait_stat)]})",
            )
            new_dataset.set_band_description(2, "Coefficient of Variation")
            new_dataset.set_band_description(3, "Area of Applicability mask")

        cog_path = (
            new_file_path.parent / f"{new_file_path.stem}_cog{new_file_path.suffix}"
        )
        log.info("Writing COG...")
        # Run command: rio cogeo create new_file_path cog_path
        subprocess.run(
            [
                "rio",
                "cogeo",
                "create",
                "--overview-resampling",
                "average",
                "--in-memory",
                str(new_file_path),
                str(cog_path),
            ],
            check=True,
        )

        if destination in ("local", "both"):
            local_dest_dir = get_processed_dir() / cfg.public.local_dir
            local_dest_dir.mkdir(parents=True, exist_ok=True)
            # shutil move
            shutil.move(cog_path, local_dest_dir / new_file_path.name)
            # copy(cog_path, dest_dir / new_file_path.name, driver="COG")

        if destination in ("sftp", "both"):
            # Upload the new file to the server
            log.info("Uploading to the server...")
            upload_file_sftp(
                cog_path,
                str(
                    Path(
                        cfg.public.sftp_dir,
                        cfg.PFT,
                        f"{cfg.model_res}deg",
                        new_file_path.name,
                    )
                ),
            )

        predict_ds.close()
        cov_dataset.close()
        aoa_dataset.close()


def cli() -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build final product from predicted trait maps."
    )

    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite existing files."
    )
    parser.add_argument("-d", "--dest", default="both", help="Destination for output.")

    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Main function."""
    syscfg = cfg[detect_system()][cfg.model_res]["build_final_product"]
    predict_dir = get_predict_dir()
    trait_set: str = "splot_gbif"

    today = datetime.today().strftime("%Y-%m-%d")

    metadata = {
        "author": "Daniel Lusk",
        "organization": "Department for Sensor-based Geoinformatics, University of Freiburg",
        "contact": "daniel.lusk@geosense.uni-freiburg.de",
        "creation_date": today,
        "version": cfg.version,
        "type": "dataset",
        "language": "en",
        "keywords": "citizen-science, plant functional traits, global, 1km, "
        "earth observation, gbif, splot, try, modis, soilgrids, vodca, worldclim",
        "license": "This dataset is provided for research purposes only. Redistribution "
        "or commercial use is prohibited without permission.",
        "rights": "This dataset is the intellectual property of the Department for "
        "Sensor-based Geoinformatics, University of Freiburg. Publication pending. "
        "Do not distribute or use for commercial purposes without express permission "
        "from the authors.",
        "PFTs": cfg.PFT.replace("_", ", "),
        "usage_notes": """This dataset contains extrapolations of trait data from the 
        TRY Trait Database matched with geotagged species observations from GBIF and 
        species abundances from sPlot. Extrapolations are the result of modeling gridded 
        trait values as a function of publicly available Earth observation datasets. All 
        model performance metrics (R2, Pearson's r, nRMSE, RMSE) are based on spatial 
        K-fold cross-validation against unseen sPlot observations only.""",
    }

    with open("reference/trait_mapping.json", "rb") as f:
        trait_mapping = json.load(f)

    with open("reference/trait_stat_mapping.json", "rb") as f:
        trait_stats = json.load(f)

    client, cluster = init_dask(
        dashboard_address=cfg.dask_dashboard,
        n_workers=syscfg.n_workers,
        threads_per_worker=syscfg.threads_per_worker,
    )

    tasks = [
        process_single_trait_map(
            trait_map, trait_set, cfg, metadata, trait_mapping, trait_stats, args.dest
        )
        for trait_map in list(predict_dir.glob("*"))
    ]
    compute(*tasks)

    close_dask(client, cluster)

    log.info("Done!")


if __name__ == "__main__":
    main()

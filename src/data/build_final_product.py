"""Combines final data products (e.g. predicted traits and CoV), adds nice metadata,
and uploads to a target destination for sharing."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import rasterio
from box import ConfigBox
from dask import compute, delayed
from dask.distributed import Client
from rasterio.enums import Resampling

from src.conf.conf import get_config
from src.conf.environment import log
from src.io.upload_sftp import upload_file_sftp
from src.utils.dataset_utils import get_processed_dir


@delayed
def process_single_trait_map(
    trait_map: Path,
    cfg: ConfigBox,
    metadata: dict,
    trait_mapping: dict,
    trait_stats: dict,
) -> None:
    """Process a single trait map."""
    log.info("Processing %s", trait_map)

    trait_id = trait_map.stem.split("_")[0].split("X")[-1]
    trait_meta = trait_mapping[trait_id]
    metadata["trait_short"] = trait_meta["short"]
    metadata["trait_long"] = trait_meta["long"]
    metadata["trait_unit"] = trait_meta["unit"]

    dataset = rasterio.open(trait_map, "r")
    bounds = dataset.bounds
    spatial_extent = f"min_x: {bounds.left}, min_y: {bounds.bottom}, max_x: {bounds.right}, max_y: {bounds.top}"

    cov_path = (
        trait_map.parents[1] / cfg.processed.cov_dir / f"{trait_map.stem}_cov.tif"
    )
    cov_dataset = rasterio.open(cov_path, "r")
    cov = cov_dataset.read(1)

    raster_meta = {
        "crs": dataset.crs,
        "resolution": dataset.res,
        "geospatial_units": "degrees",
        "grid_coordinate_system": "WGS 84",
        "spatial_extent": spatial_extent,
    }

    # Read data from the original file
    data = dataset.read(1)
    new_profile = dataset.profile.copy()
    new_profile.update(count=2)
    new_profile.update(compress="ZSTD")

    new_tags = dataset.tags().copy()

    for key, value in metadata.items():
        new_tags[key] = value

    for key, value in raster_meta.items():
        new_tags[key] = value

    # Define new file path
    with tempfile.TemporaryDirectory() as temp_dir:
        new_file_path = Path(
            temp_dir,
            f"{trait_map.stem}_{cfg.PFT}_{cfg.model_res}_deg{trait_map.suffix}",
        )

        # Create a new file with updated metadata and original data
        with rasterio.open(
            new_file_path,
            "w",
            **new_profile,
        ) as new_dataset:
            new_dataset.update_tags(**new_tags)
            new_dataset.write(data, 1)
            new_dataset.write(cov, 2)

            new_dataset.set_band_description(
                1,
                f"{trait_meta['short']} ({trait_stats[str(cfg.datasets.Y.trait_stat)]})",
            )
            new_dataset.set_band_description(2, "Coefficient of variation")

            factors = [2, 4, 8, 16]

            if dataset.overviews(1):
                factors = dataset.overviews(1)

            new_dataset.build_overviews(factors, Resampling.average)
            new_dataset.update_tags(ns="rio_overview", resampling="average")

        # Upload the new file to the server
        log.info("Uploading to the server...")
        upload_file_sftp(
            new_file_path,
            str(
                Path(
                    cfg.public.dir,
                    cfg.PFT,
                    f"{cfg.model_res}deg",
                    new_file_path.name,
                )
            ),
        )

        dataset.close()
        cov_dataset.close()


def main(cfg: ConfigBox = get_config()) -> None:
    """Main function."""
    log.info("Adding metadata to predicted trait rasters...")
    processed_dir = get_processed_dir(cfg) / cfg.processed.predict_dir

    today = datetime.today().strftime("%Y-%m-%d")

    # TODO: #10 Add full metadata @dluks
    metadata = {
        "author": "Daniel Lusk",
        "organization": "Department for Sensor-based Geoinformatics, University of Freiburg",
        "contact": "Daniel Lusk <daniel.lusk@geosense.uni-freiburg.de>",
        "creation_date": today,
        "version": cfg.version,
        "type": "dataset",
        "language": "en",
        "keywords": "citizen-science, plant functional traits, global, 1km, "
        "Earth observation",
        "license": "This dataset is provided for research purposes only. Redistribution "
        "or commercial use is prohibited without permission.",
        "rights": "This dataset is the intellectual property of the Department for "
        "Sensor-based Geoinformatics, University of Freiburg. Publication pending. "
        "Do not distribute or use for commercial purposes without express permission "
        "from the authors.",
        "PFTs": cfg.PFT.replace("_", ", "),
    }

    with open("reference/trait_mapping.json", "rb") as f:
        trait_mapping = json.load(f)

    with open("reference/trait_stat_mapping.json", "rb") as f:
        trait_stats = json.load(f)

    with Client(
        dashboard_address=cfg.dask_dashboard, n_workers=8, threads_per_worker=1
    ):
        tasks = [
            process_single_trait_map(
                trait_map, cfg, metadata, trait_mapping, trait_stats
            )
            for trait_map in processed_dir.glob("*.tif")
        ]
        compute(*tasks)

    log.info("Done!")


if __name__ == "__main__":
    main()

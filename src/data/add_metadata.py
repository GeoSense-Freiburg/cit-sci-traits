"""Adds metadata to predicted trait rasters."""

import json
import os
from datetime import datetime
from pathlib import Path

import paramiko
import rasterio
from box import ConfigBox
from rasterio.enums import Resampling

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import get_processed_dir


def upload_file_sftp(local_file_path: str | Path, remote_path: str):
    """Upload a file to a remote server using SFTP."""
    sftp_server = os.environ["SFTP_SERVER"]
    port = int(os.environ["SFTP_PORT"])  # Default SFTP port
    username = os.environ["SFTP_USER"]
    ssh_key = paramiko.Ed25519Key.from_private_key_file(os.environ["SSH_KEY_PATH"])

    # Initialize SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the server
        ssh.connect(sftp_server, port=port, username=username, pkey=ssh_key)

        # Initialize SFTP session
        sftp = ssh.open_sftp()

        # Upload file
        sftp.put(local_file_path, remote_path)
        log.info("Successfully uploaded %s to %s", local_file_path, remote_path)

        # Close SFTP session and SSH connection
        sftp.close()
        ssh.close()
    except Exception as e:
        log(f"Failed to upload file: {e}")
        ssh.close()


def main(cfg: ConfigBox = get_config()) -> None:
    """Main function."""
    log.info("Adding metadata to predicted trait rasters...")
    processed_dir = get_processed_dir(cfg) / cfg.processed.predict_dir

    today = datetime.today().strftime("%Y-%m-%d")

    metadata = {
        "author": "Daniel Lusk <daniel.lusk@geosense.uni-freiburg.de>",
        "organization": "GeoSense Lab, University of Freiburg",
        "creation_date": today,
        "version": cfg.version,
        "PFTs": cfg.PFT.replace("_", ", "),
    }

    with open("reference/trait_mapping.json", "rb") as f:
        trait_mapping = json.load(f)

    for trait_map in processed_dir.glob("*.tif"):
        log.info("Processing %s", trait_map)

        trait_id = trait_map.stem.split("_")[0].split("X")[-1]
        trait_meta = trait_mapping[trait_id]
        metadata["trait_short"] = trait_meta["short"]
        metadata["trait_long"] = trait_meta["long"]
        metadata["trait_unit"] = trait_meta["unit"]

        dataset = rasterio.open(trait_map, "r")

        raster_meta = {
            "crs": dataset.crs,
            "resolution": dataset.res,
            "geospatial_units": "degrees",
            "grid_coordinate_system": "WGS 84",
        }

        # Read data from the original file
        data = dataset.read()
        new_profile = dataset.profile.copy()
        new_tags = dataset.tags().copy()

        for key, value in metadata.items():
            new_tags[key] = value

        for key, value in raster_meta.items():
            new_tags[key] = value

        # Define new file path
        new_file_path = Path(
            "tmp", f"{trait_map.stem}_{cfg.PFT}_{cfg.model_res}_deg{trait_map.suffix}"
        )
        new_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a new file with updated metadata and original data
        with rasterio.open(
            new_file_path,
            "w",
            **new_profile,
        ) as new_dataset:
            new_dataset.update_tags(**new_tags)
            overviews = dataset.overviews(
                1
            )  # Assuming band 1 as a reference; adjust if necessary
            if overviews:
                # Recreate overviews on the new dataset
                new_dataset.build_overviews(overviews, resampling=Resampling.nearest)
                new_dataset.update_tags(ns="rio_overview", resampling="nearest")
            new_dataset.write(data)

        # Upload the new file to the server
        log.info("Uploading to the server...")
        upload_file_sftp(new_file_path, str(Path(cfg.public.dir, new_file_path.name)))

        # Remove the temporary file
        new_file_path.unlink()

        log.info("Updated metadata for %s", trait_map)

    # Remove the temporary directory
    Path("tmp").rmdir()

    log.info("Done!")


if __name__ == "__main__":
    main()

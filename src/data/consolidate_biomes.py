from pathlib import Path

import numpy as np
import rasterio
from box import ConfigBox
from rasterio.warp import Resampling, calculate_default_transform, reproject

from src.conf.conf import get_config
from src.conf.environment import log


def reclassify_biomes(
    input_raster: str | Path,
    output_raster: str | Path,
    reclassification_map: dict,
    crs: str = "EPSG:6933",
) -> None:
    """
    Reclassifies a raster file based on the biome_reclassification mapping.

    Parameters:
        input_raster (str): Path to the input raster file.
        output_raster (str): Path to save the reclassified raster file.
    """
    # Open the input raster
    log.info("Reading raster file: %s", input_raster)
    with rasterio.open(input_raster) as src:
        # Read the data as a NumPy array
        data = src.read(1)
        profile = src.profile

        # Reclassify the raster data
        log.info("Reclassifying biomes...")
        reclassified_data = np.copy(data)
        for original_biome, new_biome in reclassification_map.items():
            reclassified_data[data == original_biome] = new_biome

        # Update the profile to write a single band

        # Write the reclassified data to a new raster file
        log.info("Writing reclassified raster file: %s", output_raster)
        dst_crs = crs
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile.update(
            dtype=rasterio.uint8,
            compress="deflate",
            count=1,
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
        )
        with rasterio.open(output_raster, "w", **profile) as dst:
            reproject(
                source=reclassified_data,
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
            # dst.write(dest, indexes=i)


def main(cfg: ConfigBox = get_config()) -> None:
    """Main function to reclassify the biomes raster file."""
    input_path = Path(cfg.raw_dir, cfg.biomes.raw_path)
    output_path = Path(cfg.interim_dir, cfg.biomes.interim_path)
    reclassification_map = dict(cfg.biomes.reclassification)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    reclassify_biomes(input_path, output_path, reclassification_map, crs=cfg.crs)


if __name__ == "__main__":
    main()

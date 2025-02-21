import subprocess
from pathlib import Path

from joblib import Parallel, delayed

from src.conf.environment import log
from src.utils.dataset_utils import get_final_fns


def reproject_final_map_for_web(f: Path, tmp_dir: str) -> Path:
    log.info(f"Reprojecting {f.name}")
    tmp_fn = Path(tmp_dir) / f.name
    subprocess.run(
        [
            "gdalwarp",
            "-multi",
            "-overwrite",
            "-r",
            "bilinear",
            "-t_srs",
            "EPSG:3857",
            "-te",
            "-20037508.34",  # west  (example bounding box)
            "-20048966.1",  # south
            "20037508.34",  # east
            "20048966.1",  # north
            "-tr",
            "1000",
            "1000",
            f,
            tmp_fn,
        ]
    )

    # Now convert to web-optimized mercator with rio-cogeo
    log.info(f"Converting {f.name} to COG")
    subprocess.run(
        [
            "rio",
            "cogeo",
            "create",
            "--web-optimized",
            "--overview-resampling",
            "average",
            "--in-memory",
            tmp_fn,
            tmp_fn,
        ]
    )

    return tmp_fn


def reproject_final_maps_for_web() -> None:
    """
    Reproject the final maps to EPSG:4326 and convert them to COGs for web viewing.
    """
    fns = list(get_final_fns())

    tmp_dir = Path("/mnt/data/dl1070/tmp")
    tmp_dir.mkdir(exist_ok=True)
    Parallel(n_jobs=5)(delayed(reproject_final_map_for_web)(f, tmp_dir) for f in fns)


if __name__ == "__main__":
    reproject_final_maps_for_web()

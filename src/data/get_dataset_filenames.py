"""Get the filenames of datasets based on the specified stage of processing."""

from pathlib import Path
from typing import Generator, List

from src.conf.conf import get_config


def get_dataset_filenames(
    datasets: List[str], stage: str
) -> Generator[Path, None, None]:
    """
    Get the filenames of datasets based on the specified stage.

    Args:
        datasets (List[str]): A list of dataset names.
        stage (str): The stage of the datasets. Must be one of 'raw', 'interim', or 'processed'.

    Yields:
        Path: The path to each dataset file.

    Raises:
        ValueError: If an invalid stage is provided.

    Returns:
        Generator[Path, None, None]: A generator that yields the filenames of the datasets.
    """
    if stage not in ["raw", "interim", "processed"]:
        raise ValueError(
            "Invalid stage. Must be one of 'raw', 'interim', or 'processed'."
        )

    conf = get_config()
    if stage == "raw":
        data_dir = Path("data/raw")
        ext = ".tif"
    elif stage == "interim":
        data_dir = Path("data/interim/eo_data", conf.model_res)
        ext = ".parquet"
    else:
        data_dir = Path("data/processed")
        ext = ".parquet"

    for dataset in datasets:
        for filename in Path(data_dir, dataset).glob(f"*{ext}"):
            yield filename


def get_traits_filenames(
    dataset: str, pft: str, trait_set: str, res: str
) -> Generator[Path, None, None]:
    if dataset not in ["gbif_spo_wolf"]:
        raise ValueError("Invalid dataset. Must be 'gbif_spo_wolf'.")

    if pft not in ["Shrub_Tree_Grass", "Shrub_Tree", "Grass"]:
        raise ValueError(
            "Invalid PFT. Must be one of 'Shrub_Tree_Grass', 'Shrub_Tree', or 'Grass'."
        )

    if not trait_set.lower() in ["gbif", "splot"]:
        raise ValueError("Invalid trait set. Must be one of 'gbif' or 'splot'.")

    trait_set = "GBIF" if trait_set.lower() == "gbif" else "sPlot"

    if res not in ["001deg", "02deg", "05deg", "02deg"]:
        raise ValueError(
            "Invalid resolution. Must be one of '001deg', '02deg', '05deg', or '02deg'."
        )

    data_dir = Path("data/raw") / dataset / pft / res

    if res in ["001deg", "02deg"]:
        data_dir = data_dir / "05_range"

    for fn in data_dir.glob(f"{trait_set}*.tif"):
        yield fn


if __name__ == "__main__":
    fns = list(
        get_traits_filenames("gbif_spo_wolf", "Shrub_Tree_Grass", "gbif", "001deg")
    )
    print("Done.")

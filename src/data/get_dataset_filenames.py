"""Get the filenames of datasets based on the specified stage of processing."""

from pathlib import Path
from typing import Dict, Generator, List

from src.conf.conf import get_config


def get_dataset_filenames(stage: str, datasets: str | List[str] | None = None) -> Dict:
    """
    Get the filenames of datasets for a given stage.

    Args:
        stage (str): The stage of the dataset. Must be one of 'raw', 'interim', or 'processed'.
        datasets (str | List[str] | None, optional): The specific datasets to retrieve
            filenames for. If None, all datasets will be considered. Defaults to None.

    Returns:
        Dict: A dictionary where the keys are dataset names and the values are lists of filenames.

    Raises:
        ValueError: If an invalid stage is provided.

    """
    cfg = get_config()

    stage_map = {
        "raw": {"path": Path(cfg.raw), "ext": ".tif"},
    }

    if stage not in stage_map:
        raise ValueError(
            "Invalid stage. Must be one of 'raw', 'interim', or 'processed'."
        )

    fns = {}
    for k, v in cfg.datasets.X.items():
        fns[k] = list(stage_map[stage]["path"].glob(f"{v}/*{stage_map[stage]['ext']}"))

    if datasets is not None:
        return {k: v for k, v in fns.items() if k in datasets}

    return fns


def get_traits_filenames(
    dataset: str, pft: str, trait_set: str, res: str
) -> Generator[Path, None, None]:
    """
    Get the filenames of traits data based on the specified dataset, PFT, trait set, and
    resolution.

    Args:
        dataset (str): The dataset name. Must be 'gbif_spo_wolf'.
        pft (str): The PFT (Plant Functional Type). Must be one of 'Shrub_Tree_Grass',
            'Shrub_Tree', or 'Grass'.
        trait_set (str): The trait set. Must be one of 'gbif' or 'splot'.
        res (str): The resolution. Must be one of '001deg', '02deg', '05deg', or '02deg'.

    Yields:
        Path: The path to the traits data file.

    Raises:
        ValueError: If the dataset, PFT, trait set, or resolution is invalid.

    Returns:
        Generator[Path, None, None]: A generator that yields the filenames of traits data.
    """
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


# if __name__ == "__main__":
# print(
#     list(
#         get_traits_filenames("gbif_spo_wolf", "Shrub_Tree_Grass", "gbif", "001deg")
#     )
# )
# pprint(get_dataset_filenames("raw", "worldclim"))

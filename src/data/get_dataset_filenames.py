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

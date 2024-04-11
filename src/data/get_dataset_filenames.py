from pathlib import Path
from typing import Generator


def get_dataset_filenames(datasets: list[str]) -> Generator[Path, None, None]:
    for dataset in datasets:
        for filename in Path(f"data/raw/{dataset}").rglob("*.tif"):
            yield filename

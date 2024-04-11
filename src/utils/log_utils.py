"""Setup logging for the project."""

import logging
import os


def setup_logger(
    name: str = "__main__", level: str | int = "WARNING"
) -> logging.Logger:
    """Setup logging for the project."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )
    log = logging.getLogger(name)
    log.setLevel(level)
    return log


def subprocess_logger(name: str = "__main__") -> logging.Logger:
    """Setup a logger for a subprocess."""
    setup_logger()
    subp_log = logging.getLogger(name)
    subp_log.setLevel(logging.INFO)
    return subp_log


def setup_file_logger(
    logger_name: str = "__main__",
    log_file: str | os.PathLike = "log.txt",
    level=logging.INFO,
):
    """Setup a file logger."""
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(file_handler)

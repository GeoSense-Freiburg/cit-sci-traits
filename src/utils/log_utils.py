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


def subprocess_logger(name, level: str | int = "INFO"):
    """
    Creates and configures a logger for subprocesses.

    Args:
        name (str): The name of the logger.
        level (str | int, optional): The logging level. Defaults to "INFO".

    Returns:
        logging.Logger: The configured logger instance.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )

    handler = logging.StreamHandler()  # Use a stream handler to write logs to stdout
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


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
    return l


def get_loggers_starting_with(s: str) -> list[str]:
    """
    Returns a list of logger names that start with the specified string.

    Args:
        s (str): The string to match the logger names with.

    Returns:
        list[str]: A list of logger names that start with the specified string.
    """
    return [
        name
        for name, logger in logging.Logger.manager.loggerDict.items()
        if name.startswith(s)
    ]


def suppress_dask_logging() -> None:
    """Suppress Dask logging."""
    dask_loggers = get_loggers_starting_with("distributed")
    for logger_name in dask_loggers:
        logging.getLogger(logger_name).setLevel("WARNING")


def set_dry_run_text(dry_run: bool) -> str:
    """Set the text to indicate dry-run."""
    return " (DRY-RUN)" if dry_run else ""

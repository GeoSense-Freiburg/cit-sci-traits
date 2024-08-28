"""Environment setup for the project."""

import os

from dotenv import find_dotenv, load_dotenv

from src.utils.log_utils import setup_logger

# Load environment variables
load_dotenv(find_dotenv(), override=True)
PROJECT_ROOT = os.environ["PROJECT_ROOT"]

os.chdir(PROJECT_ROOT)

# Setup logger
log = setup_logger(__name__, "INFO")


def activate_env() -> None:
    """Setup the environment for the project."""

    load_dotenv(find_dotenv(), override=True)
    project_root = os.environ["PROJECT_ROOT"]
    detect_system()
    log.info("Project root: %s", project_root)


def detect_system() -> str:
    """Detect the system environment."""
    system = os.environ.get("SYSTEM", None)
    if system is None:
        raise ValueError("SYSTEM environment variable not set.")
    log.info("Detected system: %s", system)
    return system

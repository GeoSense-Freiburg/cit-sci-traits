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

import argparse

from src.utils.gee_utils import transfer_gs_assets_to_gee


def cli() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transfer Google Cloud Storage assets to Google Earth Engine."
    )
    parser.add_argument(
        "--gs_bucket",
        type=str,
        help="The Google Cloud Storage bucket. E.g. 'global-traits'.",
    )
    parser.add_argument(
        "--gee_project",
        type=str,
        help="The Google Earth Engine project. E.g. 'projects/global-traits'.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace = cli()) -> None:
    """Transfer Google Cloud Storage assets to Google Earth Engine."""
    transfer_gs_assets_to_gee(args.gs_bucket, args.gee_project)


if __name__ == "__main__":
    main()

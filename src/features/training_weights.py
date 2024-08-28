"""Store training weights for each y observation based on their source. Only used when
y is sourced from both GBIF and sPlot data.

E.g. When y is sourced from both GBIF and sPlot data, sPlot data is given preference. In
these instances, sPlot-derived observations will be given one weight (usually higher),
while GBIF-derived observations will be given another weight (usually lower). This is
because GBIF observations usually greatly outnumber sPlot observations, and we may want
to give sPlot observations more weight in the model.
"""

import argparse
from box import ConfigBox
from dask.distributed import Client
import pandas as pd
from src.conf.conf import get_config
from src.utils.dataset_utils import (
    get_trait_map_fns,
    get_weights_fn,
    group_y_fns,
    load_rasters_parallel,
)
from src.conf.environment import log


def cli() -> argparse.Namespace:
    """
    Parse command line arguments for featurizing training data.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Featurize training data")
    parser.add_argument(
        "-n",
        "--nchunks",
        type=int,
        default=9,
        help="Number of chunks to split data into",
    )
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Main function"""
    with Client(dashboard_address=cfg.dask_dashboard):
        # Get trait map filenames. We only need the first two because all traits share
        # the spatial coverage
        trait_map_fns = get_trait_map_fns("interim")[:2]

        log.info("Loading Y data...")
        trait_map_fns_grouped = group_y_fns(trait_map_fns)
        y_sources_ds = load_rasters_parallel(
            trait_map_fns_grouped,
            cfg.datasets.Y.trait_stat,
            args.nchunks,
            ml_set="y_source",
        )

        # Convert to Dask DataFrame and drop empty pixels
        df = (
            y_sources_ds.to_dask_dataframe()
            .drop(columns=["band", "spatial_ref"])
            .pipe(
                lambda _ddf: _ddf.dropna(
                    how="all", subset=_ddf.columns.difference(["x", "y"])
                )
            )
            .compute()
            .reset_index(drop=True)
            # .set_index(["y", "x"])
        )

        trait_col = df.columns.difference(["x", "y"]).to_list()[0]
        proportion = df[trait_col].value_counts(normalize=True)

        if len(proportion) == 1:
            df["weights"] = 1.0
        else:
            weights = pd.Series(
                {
                    "s": (
                        1.0
                        if cfg.train.weights.method == "auto"
                        else cfg.train.weights.splot
                    )
                }
            )

            # Calculate the weight for 'g' based on the proportion of 's' to 'g'
            weights["g"] = (
                proportion["s"] / proportion["g"]
                if cfg.train.weights.method == "auto"
                else cfg.train.weights.gbif
            )

        log.info("Assigning and saving weights...")
        (
            df.assign(weights=lambda _df: df[trait_col].map(weights))
            .drop(columns=trait_col)
            .set_index(["y", "x"])
            .astype({"weights": "category"})
            .to_parquet(get_weights_fn(cfg), index=True)
        )


if __name__ == "__main__":
    main()

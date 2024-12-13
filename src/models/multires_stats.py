"""Updates (or creates) model performance and feature importance data for all trait models."""

import argparse
import collections
import shutil
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import (
    get_all_fi,
    get_all_fi_fn,
    get_all_model_perf,
    get_all_model_perf_fn,
    get_all_trait_models,
    get_feature_importance,
    get_model_performance,
)


def get_all_model_perf_df(debug: bool = False) -> pd.DataFrame:
    """Get the model performance results DataFrame. If none exists yet, create it."""
    results = get_all_model_perf(debug=debug)
    if results.empty:
        results_cols = {
            "pft": str,
            "resolution": str,
            "trait_id": str,
            "trait_set": str,
            "automl": bool,
            "model_arch": str,
            "run_id": str,
            "r2": float,
            "pearsonr": float,
            "pearsonr_wt": float,
            "root_mean_squared_error": float,
            "norm_root_mean_squared_error": float,
            "mean_squared_error": float,
            "mean_absolute_error": float,
            "median_absolute_error": float,
            "transform": str,
        }
        results = pd.DataFrame(columns=[str(k) for k in results_cols]).astype(
            results_cols
        )
    return results


def get_all_fi_df(debug: bool = False) -> pd.DataFrame:
    """Get the feature importance results DataFrame. If none exists yet, create it."""
    fi = get_all_fi(debug=debug)
    if fi.empty:
        fi_cols = {
            "pft": str,
            "resolution": str,
            "trait_id": str,
            "trait_set": str,
            "automl": bool,
            "model_arch": str,
            "run_id": str,
            "feature": str,
            "agg": str,
            "importance": float,
            "stddev": float,
            "p_value": float,
            "n": int,
            "p99_high": float,
            "p99_low": float,
            "dataset": str,
        }
        fi = pd.DataFrame(columns=[str(k) for k in fi_cols]).astype(fi_cols)
    return fi


def update_model_perf(
    model_dir: Path, df: pd.DataFrame, config: ConfigBox
) -> pd.DataFrame:
    """
    Update model performance data.

    This function retrieves model performance metrics for a specific trait and
    trait set, augments the data with additional configuration details, and
    concatenates it with an existing DataFrame. Duplicate rows are removed
    from the resulting DataFrame.

    Args:
        model_dir (Path): The directory path of the model, used to extract
                          trait_id and trait_set.
        df (pd.DataFrame): The existing DataFrame to which the new performance
                           data will be appended.
        config (ConfigBox): Configuration object containing various settings
                            and parameters.

    Returns:
        pd.DataFrame: A DataFrame containing the combined performance data
                      with duplicates removed.
    """
    trait_id = model_dir.parents[2].name
    trait_set = model_dir.name
    trait_df = get_model_performance(trait_id, trait_set).assign(
        pft=config.PFT,
        resolution=config.model_res,
        trait_id=trait_id,
        automl=config.train.arch == "autogluon",
        model_arch=config[config.train.arch].included_model_types[0],
        run_id=model_dir.parent.name,
        trait_set=trait_set,
    )
    # If any columns exist on df but not on trait_df, add them to trait_df
    for col in df.columns:
        if col not in trait_df.columns:
            trait_df[col] = None

    trait_df = trait_df[df.columns]

    return pd.concat([df, trait_df], ignore_index=True).drop_duplicates(
        subset=["pft", "resolution", "run_id", "trait_set", "transform"], keep="last"
    )


def update_fi(model_dir: Path, df: pd.DataFrame, config: ConfigBox) -> pd.DataFrame:
    """
    Update feature importance data.

    This function retrieves feature importance metrics for a specific trait and
    trait set, augments the data with additional configuration details, and
    concatenates it with an existing DataFrame. Duplicate rows are removed
    from the resulting DataFrame.

    Args:
        fi (pd.DataFrame): The existing DataFrame to which the new feature
                           importance data will be appended.
        model_dir (Path): The directory path of the model, used to extract
                          trait_id and trait_set.
        config (ConfigBox): Configuration object containing various settings
                            and parameters.

    Returns:
        pd.DataFrame: A DataFrame containing the combined feature importance
                      data with duplicates removed.
    """

    def _stack_fi(_df: pd.DataFrame) -> pd.DataFrame:
        return (
            _df.stack(future_stack=True)
            .reset_index()
            .rename(columns={"index": "feature", "level_1": "agg"})
        )

    def _add_dataset(_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a column to the feature importance DataFrame indicating the dataset the feature
        belongs to.
        """
        _df = _df.copy()

        feat_ds_map = {
            "canopy_height": {"startswith": True, "match": "ETH"},
            "soilgrids": {
                "startswith": False,
                "match": "cm_mean",
            },
            "modis": {"startswith": True, "match": "sur_refl"},
            "vodca": {"startswith": True, "match": "vodca"},
            "worldclim": {"startswith": True, "match": "wc2.1"},
        }

        for ds, v in feat_ds_map.items():
            if v["startswith"]:
                _df.loc[_df.feature.str.startswith(v["match"]), "dataset"] = ds
            else:
                _df.loc[_df.feature.str.contains(v["match"]), "dataset"] = ds

        return _df

    trait_id = model_dir.parents[2].name
    trait_set = model_dir.name
    trait_fi = (
        get_feature_importance(trait_id, trait_set)
        .pipe(_stack_fi)
        .pipe(_add_dataset)
        .assign(
            pft=config.PFT,
            resolution=config.model_res,
            trait_id=trait_id,
            automl=config.train.arch == "autogluon",
            model_arch=config[config.train.arch].included_model_types[0],
            run_id=model_dir.parent.name,
            trait_set=trait_set,
        )
    )[df.columns]

    return pd.concat([df, trait_fi], ignore_index=True).drop_duplicates(
        subset=["run_id"]
    )


def map_to_trait_dfs(
    fx: Callable, df: pd.DataFrame, cfg: ConfigBox, out: Path | None = None
) -> pd.DataFrame:
    """
    Applies a given function to all trait model paths and concatenates the results into a
    single DataFrame.

    Parameters:
    fx (Callable): The function to apply to each trait model.
    df (pd.DataFrame): The input DataFrame to be processed.
    cfg (ConfigBox): Configuration settings required by the function.
    out (Path | None, optional): The file path to save the resulting DataFrame as a
        parquet file. Defaults to None.

    Returns:
    pd.DataFrame: The concatenated DataFrame after applying the function to all trait
        models.
    """
    df = pd.concat(
        list(map(partial(fx, df=df, config=cfg), get_all_trait_models())),
        ignore_index=True,
    ).drop_duplicates()

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)

    return df


def copy_cv_obs_pred(
    model_dir: Path, config: ConfigBox, overwrite: bool = False
) -> None:
    """Copy CV obs. vs pred. data to the all results directory."""
    obs_pred = model_dir / "cv_obs_vs_pred.parquet"
    out_dir = (
        Path(config.analysis.dir)
        / config.PFT
        / config.model_res
        / model_dir.parents[2].name
        / model_dir.name
        / model_dir.parent.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fn = out_dir / obs_pred.name
    if out_fn.exists() and not overwrite:
        log.info("File already exists: %s", out_fn)
        return
    if obs_pred.exists():
        log.info("Copying %s -> %s", obs_pred, out_dir)
        temp_out_fn = out_fn.with_suffix(out_fn.suffix + ".bak")
        if out_fn.exists():
            out_fn.rename(temp_out_fn)
        try:
            shutil.copy(obs_pred, out_fn)
            if temp_out_fn.exists():
                temp_out_fn.unlink()
        except Exception as e:
            if temp_out_fn.exists():
                temp_out_fn.rename(out_fn)
            log.error("Failed to copy file: %s", e)
            raise
    return


def identity(x: Any) -> Any:
    """Return the input argument."""
    return x


def consume(iterator: Iterable, use_multiprocessing: bool = False) -> None:
    """Consume an iterator and return None. Optionally use multiprocessing."""
    if use_multiprocessing:
        with Pool(cpu_count()) as pool:
            pool.map(identity, iterator)
    else:
        collections.deque(iterator, maxlen=0)


def cli() -> argparse.Namespace:
    """Simple CLI"""
    parser = argparse.ArgumentParser(
        description="Update model performance and feature importance data for all trait models."
    )
    parser.add_argument(
        "-m", "--model-perf", action="store_true", help="Update model performance data."
    )
    parser.add_argument(
        "-f",
        "--feature-importance",
        action="store_true",
        help="Update feature importance data.",
    )
    parser.add_argument(
        "-c", "--cv-obs-pred", action="store_true", help="Copy CV obs. vs pred. data."
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite data."
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    return parser.parse_args()


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """
    Main function to aggregate and update model performance and feature importance data
    for all models across traits and trait sets.

    Probably went a little overboard with the abstraction/composition here.
    """
    if args.model_perf:
        log.info("Updating model performance...")
        map_to_trait_dfs(
            update_model_perf,
            get_all_model_perf_df(debug=args.debug),
            cfg,
            get_all_model_perf_fn(cfg, debug=args.debug),
        )

    if args.feature_importance:
        log.info("Updating feature importance...")
        map_to_trait_dfs(
            update_fi,
            get_all_fi_df(debug=args.debug),
            cfg,
            get_all_fi_fn(cfg, debug=args.debug),
        )

    if args.cv_obs_pred:
        log.info("Copying CV Obs. vs Pred....")
        consume(
            iterator=map(partial(copy_cv_obs_pred, config=cfg), get_all_trait_models()),
            use_multiprocessing=True,
        )

    log.info("Done!")


if __name__ == "__main__":
    main()

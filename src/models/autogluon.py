"""Train a set of AutoGluon models using the given configuration."""

import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import dask.dataframe as dd
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import (
    get_cv_splits_dir,
    get_predict_imputed_fn,
    get_predict_mask_fn,
    get_trait_models_dir,
    get_y_fn,
)
from src.utils.df_utils import pipe_log
from src.utils.log_utils import set_dry_run_text, suppress_dask_logging
from src.utils.training_utils import assign_weights, filter_trait_set


@dataclass
class TrainOptions:
    """Configuration for training AutoGluon models."""

    sample: float
    debug: bool
    resume: bool
    dry_run: bool
    cfg: ConfigBox = get_config()


@dataclass
class TraitSetInfo:
    """Configuration for training a single trait set for a single trait."""

    trait_set: str
    trait_name: str
    training_dir: Path
    cfg: ConfigBox = get_config()

    @property
    def cv_dir(self) -> Path:
        """Directory where cross-validation models are stored."""
        return self.training_dir / "cv"

    @property
    def cv_eval_results(self) -> Path:
        """Path to the cross-validation evaluation results."""
        return self.training_dir / self.cfg.train.eval_results

    @property
    def cv_feature_importance(self) -> Path:
        """Path to the cross-validation feature importance results."""
        return self.training_dir / self.cfg.train.feature_importance

    @property
    def full_model(self) -> Path:
        """Directory where the full model is stored."""
        return self.training_dir / "full_model"

    def cv_fold_complete_flag(self, fold: int) -> Path:
        """Flag to indicate if the cross-validation evaluation results are complete for
        a given fold."""
        return self.training_dir / "cv" / f"cv_fold_{fold}_complete.flag"

    def mark_cv_fold_complete(self, fold: int) -> None:
        """Mark the cross-validation evaluation results as complete for a given fold."""
        self.cv_fold_complete_flag(fold).touch()

    @property
    def cv_complete_flag(self) -> Path:
        """Flag to indicate if the cross-validation evaluation results are complete."""
        return self.training_dir / "cv_complete.flag"

    def mark_cv_complete(self) -> None:
        """Mark the cross-validation evaluation results as complete."""
        self.cv_complete_flag.touch()

    def full_model_complete_flag(self) -> Path:
        """Flag to indicate if the full model has been trained."""
        return self.training_dir / "full_model_complete.flag"

    def mark_full_model_complete(self) -> None:
        """Mark the full model as complete."""
        self.full_model_complete_flag().touch()

    @property
    def is_cv_complete(self) -> bool:
        """Check if the cross-validation evaluation results are complete."""
        return self.cv_complete_flag.exists()

    @property
    def is_full_model_complete(self) -> bool:
        """Check if the full model has been trained."""
        return self.full_model_complete_flag().exists()

    def get_last_complete_fold_id(self) -> int | None:
        """Get the ID of the last fold for which the cross-validation evaluation results
        are complete. If no folds are complete, return None."""
        complete_folds = list(self.cv_dir.glob("cv_fold_*_complete.flag"))
        if not complete_folds:
            return None

        return max(int(f.name.split("_")[2]) for f in complete_folds)


class TraitTrainer:
    """Train AutoGluon models for a single trait using the given configuration."""

    def __init__(self, xy: pd.DataFrame, trait_name: str, opts: TrainOptions):
        """Initialize the trait trainer."""
        self.xy = xy
        self.trait_name = trait_name
        self.opts = opts
        self.dry_run_text = set_dry_run_text(opts.dry_run)

        if opts.sample < 1.0:
            self._log_subsampling()
            self.xy = self._sample_xy() if not opts.dry_run else self.xy

        self.runs_dir: Path = (
            get_trait_models_dir(self.trait_name) / "debug"
            if opts.debug
            else get_trait_models_dir(self.trait_name)
        )

        sorted_runs = sorted(
            [run for run in self.runs_dir.glob("*") if "tmp" not in run.name],
            reverse=True,
        )

        if not sorted_runs:
            log.warning("No prior runs found in %s. Creating new run...", self.runs_dir)
            self.last_run: Path = self.runs_dir / now()
            self.current_run: Path = self.last_run
        else:
            self.last_run: Path = sorted_runs[0]
            self.current_run: Path = (
                self.last_run if opts.resume else self.runs_dir / now()
            )

    def _sample_xy(self) -> pd.DataFrame:
        """Sample the input data for quick prototyping."""
        return self.xy.sample(
            frac=self.opts.sample, random_state=self.opts.cfg.random_seed
        )

    def _log_is_trained_full(self, trait_set: str) -> None:
        """Log that all models (CV and full) have already been trained for the given
        trait set."""
        log.info(
            "All models for %s already trained for %s trait set. Skipping...%s",
            self.trait_name,
            trait_set,
            self.dry_run_text,
        )

    def _log_is_trained_cv(self, trait_set: str) -> None:
        """Log that CV models have already been trained for the given trait set."""
        log.info(
            "CV models for %s already trained for %s trait set. Skipping directly to "
            "full model training...%s",
            self.trait_name,
            trait_set.upper(),
            self.dry_run_text,
        )

    def _log_is_trained_partial_cv(self, trait_set: str) -> None:
        """Log that the CV model training is only partially complete for the given trait
        set."""
        log.info(
            "CV training for %s not complete for %s trait set. Resuming training...%s",
            self.trait_name,
            trait_set.upper(),
            self.dry_run_text,
        )

    def _log_training(self, trait_set: str) -> None:
        """Log that the model is being trained for the given trait set."""
        log.info(
            "Training model for %s with %s trait set...%s",
            self.trait_name,
            trait_set.upper(),
            self.dry_run_text,
        )

    def _log_subsampling(self) -> None:
        """Log that the data is being subsampled."""
        log.info(
            "Subsampling %i%% of the data...%s",
            self.opts.sample * 100,
            self.dry_run_text,
        )

    def _log_full_training(self, trait_set: str) -> None:
        """Log that the full model is being trained."""
        log.info(
            "Training model on all data for trait %s from %s trait set...%s",
            self.trait_name,
            trait_set.upper(),
            self.dry_run_text,
        )

    @staticmethod
    def _aggregate_results(cv_dir: Path, target: str) -> pd.DataFrame:
        return (
            pd.concat(
                [
                    pd.read_csv(fold_model_path / target, index_col=0)
                    for fold_model_path in cv_dir.glob("fold_*")
                ],
            )
            .drop(columns=["fold"])
            .reset_index(names="index")
            .groupby("index")
            .agg(["mean", "std"])
        )

    # We set this to avoid a bug in LightGBM when used with GPU.
    # See https://github.com/microsoft/LightGBM/issues/3679

    def _train_full_model(self, ts_info: TraitSetInfo):
        train_full = TabularDataset(
            self.xy.pipe(filter_trait_set, ts_info.trait_set)
            .dropna(subset=[self.trait_name])
            .pipe(assign_weights, w_gbif=self.opts.cfg.train.weights.gbif)
            .drop(columns=["x", "y", "source", "fold"])
        )

        HYPERPARAMS: dict = {
            "GBM": {
                "device": "cpu",
                # "ag_args_fit": {
                #     "num_gpus": self.opts.cfg.autogluon.num_gpus // 4,
                #     "num_cpus": self.opts.cfg.autogluon.num_cpus // 4,
                # },
            }
        }

        predictor = TabularPredictor(
            label=ts_info.trait_name,
            sample_weight="weights",  # pyright: ignore[reportArgumentType]
            path=str(ts_info.full_model),
        ).fit(
            train_full,
            included_model_types=self.opts.cfg.autogluon.included_model_types,
            num_gpus=self.opts.cfg.autogluon.num_gpus,
            num_cpus=self.opts.cfg.autogluon.num_cpus,
            presets=self.opts.cfg.autogluon.presets,
            time_limit=self.opts.cfg.autogluon.full_fit_time_limit,
            save_bag_folds=self.opts.cfg.autogluon.save_bag_folds,
            hyperparameters=HYPERPARAMS,
            feature_prune_kwargs={},
        )

        ts_info.mark_full_model_complete()
        predictor.save_space()

    def _train_fold(self, fold_id: int, cv_dir: Path, trait_set: str) -> None:
        log.info("Training model for fold %d...", fold_id)
        fold_model_path = cv_dir / f"fold_{fold_id}"

        HYPERPARAMS: dict = {
            "GBM": {
                "device": "cpu",
                # "ag_args_fit": {
                #     "num_gpus": self.opts.cfg.autogluon.num_gpus // 4,
                #     "num_cpus": self.opts.cfg.autogluon.num_cpus // 4,
                # },
            }
        }

        train = TabularDataset(
            self.xy[self.xy["fold"] != fold_id]
            .pipe(filter_trait_set, trait_set)
            .dropna(subset=[self.trait_name])
            .pipe(assign_weights, w_gbif=self.opts.cfg.train.weights.gbif)
            .drop(columns=["x", "y", "source", "fold"])
            .reset_index(drop=True)
        )
        val = TabularDataset(
            self.xy[self.xy["fold"] == fold_id]
            .query("source == 's'")
            .dropna(subset=[self.trait_name])
            .assign(weights=1.0)
            .drop(columns=["x", "y", "source", "fold"])
            .reset_index(drop=True)
        )

        try:
            predictor = TabularPredictor(
                label=self.trait_name,
                sample_weight="weights",  # pyright: ignore[reportArgumentType]
                path=str(fold_model_path),
            ).fit(
                train,
                included_model_types=self.opts.cfg.autogluon.included_model_types,
                num_gpus=self.opts.cfg.autogluon.num_gpus,
                num_cpus=self.opts.cfg.autogluon.num_cpus,
                presets=self.opts.cfg.autogluon.presets,
                time_limit=self.opts.cfg.autogluon.cv_fit_time_limit,
                save_bag_folds=self.opts.cfg.autogluon.save_bag_folds,
                hyperparameters=HYPERPARAMS,
                # ds_args={"validation_procedure": "holdout", "holdout_data": val},
                feature_prune_kwargs={},
            )

            if self.opts.cfg.autogluon.feature_importance:
                log.info("Calculating feature importance...")
                features = predictor.feature_metadata_in.get_features()
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
                # Generate a list of tuples of (dataset, [features]) for each dataset
                datasets = []
                for ds, ds_info in feat_ds_map.items():
                    if ds_info["startswith"]:
                        ds_feats = [
                            feat
                            for feat in features
                            if feat.startswith(ds_info["match"])
                        ]
                    else:
                        ds_feats = [
                            feat for feat in features if feat.endswith(ds_info["match"])
                        ]
                    datasets.append((ds, ds_feats))

                # Now add all features as well
                datasets += features

                feature_importance = predictor.feature_importance(
                    val,
                    features=datasets,
                    time_limit=self.opts.cfg.autogluon.FI_time_limit,
                    num_shuffle_sets=self.opts.cfg.autogluon.FI_num_shuffle_sets,
                ).assign(fold=fold_id)

                feature_importance.to_csv(
                    fold_model_path / self.opts.cfg.train.feature_importance
                )

            log.info(
                "Evaluating fold model (%s/%s)...",
                fold_id + 1,
                self.opts.cfg.train.cv_splits.n_splits,
            )
            eval_results = predictor.evaluate(
                val, auxiliary_metrics=True, detailed_report=True
            )

            # Normalize RMSE by the 99th percentile - 1st percentile range of the target
            norm_factor = val[self.trait_name].quantile(0.99) - val[
                self.trait_name
            ].quantile(0.01)
            eval_results["norm_root_mean_squared_error"] = (
                eval_results["root_mean_squared_error"] / norm_factor
            )

            pd.DataFrame({col: [val] for col, val in eval_results.items()}).assign(
                fold=fold_id
            ).to_csv(fold_model_path / self.opts.cfg.train.eval_results)

            predictor.save_space()

        except ValueError as e:
            log.error("Error training model: %s", e)
            raise

    def _aggregate_cv_results(self, cv_dir: Path, training_dir: Path):
        log.info("Aggregating evaluation results...")
        eval_df = self._aggregate_results(cv_dir, self.opts.cfg.train.eval_results)
        eval_df.to_csv(training_dir / self.opts.cfg.train.eval_results)

        log.info("Aggregating feature importance...")
        fi_df = self._aggregate_results(cv_dir, self.opts.cfg.train.feature_importance)
        fi_df.to_csv(training_dir / self.opts.cfg.train.feature_importance)

    def _train_models_cv(self, ts_info: TraitSetInfo) -> None:
        ts_info.cv_dir.mkdir(parents=True, exist_ok=True)

        last_complete_fold = ts_info.get_last_complete_fold_id()
        starting_fold = last_complete_fold + 1 if last_complete_fold is not None else 0

        for i in range(starting_fold, max(self.xy["fold"].unique()) + 1):
            self._train_fold(i, ts_info.cv_dir, ts_info.trait_set)
            ts_info.mark_cv_fold_complete(i)

        self._aggregate_cv_results(ts_info.cv_dir, ts_info.training_dir)
        ts_info.mark_cv_complete()

    def _train_trait_set(self, trait_set: str) -> None:
        """Train AutoGluon models for a single trait using the given configuration."""
        dry_run = self.opts.dry_run

        ts_info = TraitSetInfo(
            trait_set,
            self.trait_name,
            self.current_run / trait_set,
        )

        if not dry_run:
            ts_info.training_dir.mkdir(parents=True, exist_ok=True)

        if ts_info.is_cv_complete and ts_info.is_full_model_complete:
            self._log_is_trained_full(trait_set)
            return

        if not ts_info.is_cv_complete:
            self._log_is_trained_partial_cv(trait_set)
            if not dry_run:
                self._train_models_cv(ts_info)
        else:
            self._log_is_trained_cv(trait_set)

        self._log_full_training(trait_set)
        if not dry_run:
            self._train_full_model(ts_info)

    def train_trait_models_all_y_sets(self) -> None:
        """Train a set of AutoGluon models for a single trait based on each trait set."""
        for trait_set in ["splot", "splot_gbif", "gbif"]:
            self._train_trait_set(trait_set)

    def train_splot(self) -> None:
        """Train AutoGluon models for the "splot" trait set."""
        self._train_trait_set("splot")

    def train_gbif(self) -> None:
        """Train AutoGluon models for the "gbif" trait set."""
        self._train_trait_set("gbif")

    def train_splot_gbif(self) -> None:
        """Train AutoGluon models for the "splot_gbif" trait set."""
        self._train_trait_set("splot_gbif")


def now() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def prep_full_xy(
    feats: dd.DataFrame,
    feats_mask: pd.DataFrame,
    labels: dd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    """
    Prepare the input data for modeling by filtering and assigning weights.

    Args:
        feats (dd.DataFrame): The input features.
        feats_mask (pd.DataFrame): The mask for filtering the features.
        labels (dd.DataFrame): The input labels.
        label_col (str): The column name of the labels.
        trait_set (str): The trait set to filter the data.

    Returns:
        pd.DataFrame: The prepared input data for modeling.
    """
    # TODO: #13 Speed up by leveraging dask for masking and merging
    log.info("Loading splits...")
    splits = (
        dd.read_parquet(get_cv_splits_dir() / f"{label_col}.parquet")
        .compute()
        .set_index(["y", "x"])
    )

    log.info("Merging splits and label data...")
    label = (
        labels[["x", "y", label_col, "source"]]
        .compute()
        .set_index(["y", "x"])
        .merge(splits, validate="m:1", right_index=True, left_index=True)
    )

    return (
        feats.compute()
        .set_index(["y", "x"])
        .pipe(pipe_log, "Masking features...")
        .mask(feats_mask)
        .pipe(pipe_log, "Merging features and label data...")
        .merge(label, validate="1:m", right_index=True, left_index=True)
        .reset_index()
    )


def load_data() -> tuple[dd.DataFrame, pd.DataFrame, dd.DataFrame]:
    """Load the input data for modeling."""
    feats = dd.read_parquet(get_predict_imputed_fn())
    feats_mask = pd.read_parquet(get_predict_mask_fn()).set_index(["y", "x"])
    labels = dd.read_parquet(get_y_fn())
    return feats, feats_mask, labels


def train_models(
    trait_sets: Iterable[str] | None = None,
    sample: float = 1.0,
    debug: bool = False,
    resume: bool = True,
    dry_run: bool = False,
) -> None:
    """Train a set of AutoGluon models for each  using the given configuration."""
    dry_run_text = set_dry_run_text(dry_run)
    suppress_dask_logging()

    train_opts = TrainOptions(sample, debug, resume, dry_run)

    log.info("Loading data...%s", dry_run_text)

    feats, feats_mask, labels = load_data()

    for label_col in labels.columns.difference(["x", "y", "source"]):
        tmp_xy_path = get_trait_models_dir(label_col) / "tmp" / "xy.parquet"

        if not tmp_xy_path.exists() and resume:
            runs = [
                run
                for run in get_trait_models_dir(label_col).glob("*")
                if run.is_dir() and "tmp" not in run.name
            ]

            if not runs:
                log.warning(
                    "No prior runs found for %s. Creating new run...%s",
                    label_col,
                    dry_run_text,
                )
            else:
                latest_run = max(
                    run
                    for run in get_trait_models_dir(label_col).glob("*")
                    if run.is_dir() and "tmp" not in run.name
                )

                completed = [
                    TraitSetInfo(
                        trait_set, label_col, latest_run / trait_set
                    ).is_full_model_complete
                    for trait_set in ["splot", "splot_gbif", "gbif"]
                ]

                if all(completed):
                    log.info(
                        "All models for %s already trained. Skipping...%s",
                        label_col,
                        dry_run_text,
                    )
                    continue

        log.info("Preparing data for %s training...%s", label_col, dry_run_text)
        if not tmp_xy_path.exists() or not resume:

            def _to_ddf(df: pd.DataFrame) -> dd.DataFrame:
                return dd.from_pandas(df, npartitions=100)

            if not dry_run:
                xy = prep_full_xy(feats, feats_mask, labels, label_col)
                log.info("Saving xy data for %s...%s", label_col, dry_run_text)
                tmp_xy_path.parent.mkdir(parents=True, exist_ok=True)
                xy.pipe(_to_ddf).to_parquet(
                    tmp_xy_path, compression="zstd", overwrite=True
                )
        else:
            log.info(
                "Found existing xy data for %s. Loading...%s", label_col, dry_run_text
            )
            if not dry_run:
                xy = dd.read_parquet(tmp_xy_path).compute().reset_index(drop=True)

        trait_trainer = TraitTrainer(xy, label_col, train_opts)

        valid_trait_sets = ("splot", "gbif", "splot_gbif")

        if trait_sets is None:
            trait_sets = valid_trait_sets

        for ts in trait_sets:
            if ts not in valid_trait_sets:
                raise ValueError(f"Invalid trait set: {ts}")

            if ts == "splot":
                trait_trainer.train_splot()
            elif ts == "gbif":
                trait_trainer.train_gbif()
            elif ts == "splot_gbif":
                trait_trainer.train_splot_gbif()

        log.info("Cleaning up...%s", dry_run_text)
        if not dry_run:
            shutil.rmtree(tmp_xy_path.parent)

    log.info("Done! \U00002705")

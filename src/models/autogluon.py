"""Train a set of AutoGluon models using the given configuration."""

import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from box import ConfigBox

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.autogluon_utils import get_best_model_ag
from src.utils.dataset_utils import (
    get_cv_splits_dir,
    get_predict_imputed_fn,
    get_predict_mask_fn,
    get_trait_models_dir,
    get_y_fn,
)
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

        return max(int(f.name.split("_")[1]) for f in complete_folds)


class TraitTrainer:
    """Train AutoGluon models for a single trait using the given configuration."""

    def __init__(self, xy: pd.DataFrame, trait_name: str, opts: TrainOptions):
        """Initialize the trait trainer."""
        self.xy = xy
        self.trait_name = trait_name
        self.opts = opts
        self.dry_run_text = set_dry_run_text(opts.dry_run)

    @property
    def runs_dir(self) -> Path:
        """Directory where models are stored for the current trait and ML architecture.
        If debug mode is enabled, the models are stored in the "debug" subdirectory."""
        return (
            get_trait_models_dir(self.trait_name) / "debug"
            if self.opts.debug
            else get_trait_models_dir(self.trait_name)
        )

    @property
    def current_run(self) -> Path:
        """The directory where the current run is (if resuming training) or will be stored."""
        if self.opts.resume:
            return self.last_run
        return self.runs_dir / now()

    @property
    def last_run(self) -> Path:
        """The most recent model in the trait directory."""
        sorted_models = sorted(list(self.runs_dir.glob("x")), reverse=True)
        if not sorted_models:
            raise FileNotFoundError(f"No models found in {self.runs_dir}")

        return sorted_models[0]

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
        log.info("Aggregating evaluation results...")
        return (
            pd.concat(
                [
                    pd.read_csv(fold_model_path / target)
                    for fold_model_path in cv_dir.glob("fold_*")
                ],
                ignore_index=True,
            )
            .groupby("fold")
            .agg(["mean", "std"])
        )

    def _train_full_model(self, ts_info: TraitSetInfo) -> TabularPredictor:
        train_full = TabularDataset(
            self.xy.pipe(filter_trait_set, ts_info.trait_set)
            .pipe(assign_weights)
            .drop(columns=["fold"])
        )

        return TabularPredictor(
            label=ts_info.trait_name,
            sample_weight=(
                "weights" if "weights" in self.xy.columns else None
            ),  # pyright: ignore[reportArgumentType]
            path=str(ts_info.full_model),
        ).fit(
            train_full,
            included_model_types=self.opts.cfg.autogluon.included_model_types,
            num_gpus=self.opts.cfg.autogluon.num_gpus,
            presets=self.opts.cfg.autogluon.presets,
            time_limit=self.opts.cfg.autogluon.time_limit,
        )

        ts_info.mark_full_model_complete()

    def _train_fold(self, fold_id: int, cv_dir: Path, trait_set: str) -> None:
        log.info("Training model for fold %d...", fold_id)
        fold_model_path = cv_dir / f"fold_{fold_id}"
        fold_model_path.mkdir(parents=True, exist_ok=True)

        train = TabularDataset(
            self.xy[self.xy["fold"] != fold_id]
            .pipe(filter_trait_set, trait_set)
            .pipe(assign_weights)
            .reset_index(drop=True)
        )
        val = TabularDataset(
            self.xy[self.xy["fold"] == fold_id]
            .query("source == 's'")
            .reset_index(drop=True)
        )

        try:
            predictor = TabularPredictor(
                label=self.trait_name,
                sample_weight=(
                    "weights" if "weights" in self.xy.columns else None
                ),  # pyright: ignore[reportArgumentType]
                path=str(fold_model_path),
            ).fit(
                train,
                included_model_types=self.opts.cfg.autogluon.included_model_types,
                num_gpus=self.opts.cfg.autogluon.num_gpus,
                presets=self.opts.cfg.autogluon.presets,
                time_limit=self.opts.cfg.autogluon.time_limit,
            )

            if self.opts.cfg.autogluon.feature_importance:
                log.info("Calculating feature importance...")
                feature_importance = predictor.feature_importance(
                    val,
                    time_limit=self.opts.cfg.autogluon.FI_time_limit,
                    num_shuffle_sets=self.opts.cfg.autogluon.FI_num_shuffle_sets,
                ).assign(fold=fold_id)

                feature_importance.to_csv(
                    fold_model_path / self.opts.cfg.train.feature_importance
                )

            log.info(
                "Evaluating model (Fold %s/%s)...",
                fold_id + 1,
                self.opts.cfg.train.cv_splits.n_splits,
            )
            eval_results = predictor.evaluate(
                val, auxiliary_metrics=True, detailed_report=True
            )
            pd.DataFrame(eval_results).assign(fold=fold_id).to_csv(
                fold_model_path / self.opts.cfg.train.eval_results
            )

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

        for i in range(starting_fold, max(self.xy["fold"].unique())):
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

        if self.opts.sample < 1.0:
            self._log_subsampling()
            self.xy = self._sample_xy() if not dry_run else self.xy

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


def is_model_already_trained(
    model_dir: Path, y_col: str, config: ConfigBox = get_config()
) -> bool:
    """Check if the model has already been trained."""

    # We know a model has been trained if the most recent model in the trait
    # directory starts with the configured preset, and if evaluation_results.csv,
    # feature_importance.csv, and leaderboard.csv are present.

    best_model = get_best_model_ag(model_dir / y_col)
    if best_model is not None and best_model.stem.startswith(config.autogluon.presets):
        if (
            Path(best_model, config.train.eval_results).exists()
            and Path(best_model, config.train.feature_importance).exists()
            and Path(best_model, config.autogluon.leaderboard).exists()
        ):
            return True

    return False


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

    def pipe_log(df: pd.DataFrame, message: str) -> pd.DataFrame:
        log.info(message)
        return df

    log.info("Merging features and label data...")
    return (
        feats.compute()
        .set_index(["y", "x"])
        .pipe(pipe_log, "Masking features...")
        .mask(feats_mask)
        .pipe(pipe_log, "Merging...")
        .merge(label, validate="1:m", right_index=True, left_index=True)
    )


def load_data() -> tuple[dd.DataFrame, pd.DataFrame, dd.DataFrame]:
    """Load the input data for modeling."""
    feats = dd.read_parquet(get_predict_imputed_fn())
    feats_mask = pd.read_parquet(get_predict_mask_fn()).set_index(["y", "x"])
    labels = dd.read_parquet(get_y_fn())
    return feats, feats_mask, labels


def train_models(
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
        log.info("Preparing data for %s training...%s", label_col, dry_run_text)

        tmp_xy_path = get_trait_models_dir(label_col) / "tmp" / "xy.parquet"
        if not tmp_xy_path.exists() or not resume:

            def _to_ddf(df: pd.DataFrame) -> dd.DataFrame:
                return dd.from_pandas(df, npartitions=100)

            if not dry_run:
                tmp_xy_path.parent.mkdir(parents=True, exist_ok=True)
                xy = prep_full_xy(feats, feats_mask, labels, label_col)
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
        trait_trainer.train_splot()
        trait_trainer.train_gbif()
        trait_trainer.train_splot_gbif()

        log.info("Cleaning up...%s", dry_run_text)
        if not dry_run:
            shutil.rmtree(tmp_xy_path.parent)

    log.info("Done! \U00002705")

"""Calculate Area of Applicability (AOA) for a given trait"""

import argparse
from pathlib import Path

import cudf
import cupy as cp
import dask.dataframe as dd
import pandas as pd
from box import ConfigBox
from cuml.metrics import pairwise_distances
from cuml.neighbors import NearestNeighbors
from distributed import Client

from src.conf.conf import get_config
from src.conf.environment import detect_system, log
from src.utils.cuda_utils import df_to_cupy
from src.utils.dask_cuda_utils import init_dask_cuda
from src.utils.dask_utils import close_dask, df_to_dd, init_dask
from src.utils.dataset_utils import (
    get_aoa_dir,
    get_latest_run,
    get_predict_imputed_fn,
    get_trait_models_dir,
    get_y_fn,
)
from src.utils.df_utils import rasterize_points
from src.utils.raster_utils import xr_to_raster
from src.utils.training_utils import assign_splits, filter_trait_set, set_yx_index
from src.utils.trait_utils import get_active_traits

CFG = get_config()


def cli() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate Area of Applicability (AOA)"
    )
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite files"
    )
    return parser.parse_args()


def load_train_data(
    y_col: str, trait_set: str, sample: int | float = 1
) -> pd.DataFrame:
    """Load the training data for the AOA analysis."""
    with Client(n_workers=20, dashboard_address=CFG.dask_dashboard):
        train = (
            pd.read_parquet(get_y_fn(), columns=["x", "y", y_col, "source"])
            .pipe(set_yx_index)
            .pipe(assign_splits, label_col=y_col)
            .groupby("fold", group_keys=False)
            .sample(frac=sample, random_state=CFG.random_seed)
            .reset_index()
            .pipe(filter_trait_set, trait_set=trait_set)
            .drop(columns=[y_col, "source"])
            .pipe(df_to_dd, npartitions=50)
            .merge(
                # Merge using inner join with the imputed predict data
                dd.read_parquet(  # pyright: ignore[reportPrivateImportUsage]
                    get_predict_imputed_fn()
                ).repartition(npartitions=200),
                how="inner",
                on=["x", "y"],
            )
            .drop(columns=["x", "y"])
            .reset_index(drop=True)
            .compute()
            .reset_index(drop=True)
        )

    return train


def load_predict_data(
    npartitions: int | None = None, sample: int | float = 1
) -> dd.DataFrame:
    """Load the imputed predict data for the AOA analysis."""
    ddf = dd.read_parquet(get_predict_imputed_fn()).sample(
        frac=sample, random_state=CFG.random_seed
    )

    if npartitions is not None:
        return ddf.repartition(npartitions=npartitions)

    return ddf


def scale_features(df: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    """Scale/standardize the features in the dataframe."""
    return (df - means) / stds


def load_feature_importance(
    columns: pd.Index, y_col: str, trait_set: str
) -> pd.DataFrame:
    """Load the feature importance and reorganize columns to match the current dataframe."""
    return (
        pd.read_csv(
            get_latest_run(get_trait_models_dir(y_col))
            / trait_set
            / CFG.train.feature_importance,
            index_col=0,
            header=[0, 1],
        )
        .sort_values(by=("importance", "mean"), ascending=False)["importance"]["mean"]
        .to_frame()
        .loc[columns]
    )


def weight_features(
    df: pd.DataFrame, fi: pd.DataFrame, dask: bool = False
) -> pd.DataFrame:
    """Weight the features in the dataframe by the feature importance."""
    if dask:
        return dd.concat([df * fi.T.values], axis=1)

    return pd.concat([df * fi.T.values], axis=1)


def scale_and_weight_train(
    df: pd.DataFrame, fi: pd.DataFrame, means: pd.Series, stds: pd.Series
) -> pd.DataFrame:
    """Scale and weight the training data."""
    folds = df[["fold"]].copy()
    df = df.drop(columns=["fold"])
    return pd.concat(
        [weight_features(scale_features(df, means, stds), fi), folds], axis=1
    )


def scale_and_weight_predict(
    df: dd.DataFrame, fi: pd.DataFrame, means: pd.Series, stds: pd.Series
) -> dd.DataFrame:
    """Scale and weight the predict data."""
    xy = df[["x", "y"]].copy()
    df = df.drop(columns=["x", "y"])
    return dd.concat(
        [
            xy,
            weight_features(
                df.map_partitions(scale_features, means, stds), fi, dask=True
            ),
        ],
        axis=1,
    )


def df_to_cupy_chunks(
    df: pd.DataFrame, chunk_size: int, n_samples: int, device_id: int
) -> list[cp.ndarray]:
    """Convert a DataFrame to a list of CuPy arrays in chunks."""
    cupy_data = df_to_cupy(df, device_id=device_id)
    return [cupy_data[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]


def average_train_distance_chunked(
    train_df: pd.DataFrame, num_chunks: int, device_ids: tuple[int, ...]
) -> float:
    """Compute the mean of the average pairwise distances for the training data using chunked pairwise distance calculations."""

    def _process_chunk(chunk1: cp.ndarray, chunk2: cp.ndarray) -> cp.ndarray:
        distances = pairwise_distances(chunk1, chunk2, metric="euclidean")
        avg_distances = cp.mean(distances)
        return avg_distances

    client, cluster = init_dask_cuda(device_ids=device_ids)

    # Define chunk size based on the number of chunks
    n_samples = train_df.shape[0]
    chunk_size = (n_samples + num_chunks - 1) // num_chunks

    chunks = df_to_cupy_chunks(train_df, chunk_size, n_samples, device_ids[0])

    # Scatter each chunk separately
    chunks = client.scatter(chunks)

    # Create Dask futures for each chunk pair
    futures = []
    for chunk1 in chunks:
        for chunk2 in chunks:
            future = client.submit(_process_chunk, chunk1, chunk2)
            futures.append(future)

    chunk_results = client.gather(futures)

    close_dask(client, cluster)

    # Compute the overall mean of the average distances
    avg_distances = cp.mean(cp.array(chunk_results))

    return avg_distances.item()


def average_train_distance(
    train_df: pd.DataFrame, batch_size: int, device_ids: tuple[int, ...]
) -> float:
    """Compute the mean of the average pairwise distances for the training data. This is
    used to normalize the DI values for the AOA analysis."""

    def _process_batch(data: cp.ndarray, start_idx: int, end_idx: int) -> cp.ndarray:
        batch_data = data[start_idx:end_idx]
        distances = pairwise_distances(data, batch_data)
        avg_distances = cp.mean(distances, axis=1)
        return avg_distances

    client, cluster = init_dask_cuda(device_ids=device_ids)

    # Convert data to CuPy for GPU processing
    cupy_data = df_to_cupy(train_df, device_ids[0])
    cupy_data = client.scatter(cupy_data, broadcast=True)

    # Define number of batches
    n_samples = train_df.shape[0]
    num_batches = (n_samples + batch_size - 1) // batch_size

    # Create Dask delayed tasks for each batch
    futures = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        future = client.submit(_process_batch, cupy_data, start_idx, end_idx)
        futures.append(future)

    batch_results = client.gather(futures)

    close_dask(client, cluster)

    # Sum the results and normalize by the number of batches
    avg_distances = cp.sum(cp.array(batch_results), axis=0) / num_batches

    # Compute the overall mean of the average distances
    return cp.mean(avg_distances).item()  # Convert from CuPy to Python float


def _train_folds_to_cupy(df: pd.DataFrame, device_id: int) -> cp.ndarray:
    with cp.cuda.Device(device_id):
        df_gpu = cudf.DataFrame.from_pandas(df)
        folds = [
            df_gpu[df_gpu["fold"] == i]
            .drop(columns=["fold"])  # pyright: ignore[reportOptionalMemberAccess]
            .to_cupy()  # pyright: ignore[reportOptionalMemberAccess]
            for i in range(5)
        ]
    return folds


def get_min_dists(input_data: cp.ndarray, ref_data: cp.ndarray) -> cp.ndarray:
    """Calculate the minimum distances between observations in one fold and all other folds."""
    nn_model = NearestNeighbors(n_neighbors=1, algorithm="brute")
    nn_model.fit(ref_data)
    distances, _ = nn_model.kneighbors(input_data)
    return distances.flatten()


def calc_di_threshold(
    df: pd.DataFrame,
    mean_distance: float,
    device_ids: tuple[int, ...],
) -> float:
    """Compute the DI threshold for the training data leveraging spatial cross-validation and
    the average pairwise distance."""
    # Separate the data into folds based on the 'fold' column
    folds = _train_folds_to_cupy(df, device_ids[0])

    client, cluster = init_dask_cuda(device_ids=device_ids[1:])
    # client.restart()

    folds_scattered = client.scatter(folds, broadcast=True)

    # Create delayed tasks for each fold-fold combination
    futures = []
    for fold_index, fold_data in enumerate(folds_scattered):
        for other_index, other_fold_data in enumerate(folds_scattered):
            if fold_index == other_index:
                continue  # Skip the current fold

            # task = delayed(calculate_fold_min_distances)(fold_data, other_fold_data)
            future = client.submit(
                get_min_dists, fold_data, other_fold_data, retries=10
            )
            futures.append(future)

    # min_distances_batches = compute(*futures)
    min_distances_batches = client.gather(futures)
    close_dask(client, cluster)

    # Combine the results from all batches
    min_distances = cp.concatenate([cp.array(dist) for dist in min_distances_batches])
    di = min_distances / mean_distance

    # Find the upper whisker threshold for the DI values (75th percentile + 1.5 * IQR)
    di_threshold = cp.percentile(di, 75) + 1.5 * cp.subtract(
        *cp.percentile(di, [75, 25])
    )

    return di_threshold.item()


def calc_di_predict(
    predict: dd.DataFrame,
    train: pd.DataFrame,
    mean_distance: float,
    di_threshold: float,
    device_ids: tuple[int, ...],
) -> pd.DataFrame:
    """Compute the DI values and AOA mask for the predict data using the training data
    and DI threshold."""
    client, cluster = init_dask_cuda(device_ids=device_ids)

    predict_gpu = predict.to_backend("cudf")
    train_gpu = dd.from_pandas(train).to_backend("cudf")

    def _compute_nearest_neighbors(
        pred_partition: cudf.DataFrame, train_df: cudf.DataFrame
    ) -> cudf.DataFrame:
        distances = get_min_dists(
            pred_partition.drop(columns=["x", "y"]).to_cupy(),
            train_df.to_cupy(),
        )
        result = cudf.DataFrame(
            {
                "x": pred_partition["x"],
                "y": pred_partition["y"],
                "distance": distances,
            },
            index=pred_partition.index,
        )
        return result

    distances = predict_gpu.map_partitions(_compute_nearest_neighbors, train_gpu)
    distances = distances.compute()

    close_dask(client, cluster)

    distances["di"] = distances["distance"] / mean_distance
    distances["aoa"] = distances["di"] > di_threshold
    return distances.drop(columns=["distance"]).to_pandas()


def calc_aoa(
    trait: str,
    trait_set: str,
    out_path: Path,
    cfg: ConfigBox,
    syscfg: ConfigBox,
    overwrite: bool = False,
) -> None:
    """Calculate the Area of Applicability (AoA) for the given trait and trait set."""
    log.info("Calculating AoA for %s using %s...", trait, trait_set)
    train_fn = out_path.parent / f"{trait}_train_{trait_set}.parquet"
    ts_cfg = syscfg[trait_set]

    if train_fn.exists() and not overwrite:
        log.info("Loading existing training data...")
        train = pd.read_parquet(train_fn)
    else:
        log.info("Generating training data...")
        train = load_train_data(y_col=trait, trait_set=trait_set)
        log.info("Writing training data to disk in case of failure...")
        train.to_parquet(train_fn, compression="zstd")

    log.info("Scaling and weighting training data...")
    train_means = train.drop(columns=["fold"]).mean()
    train_stds = train.drop(columns=["fold"]).std()

    fi = load_feature_importance(train.drop(columns=["fold"]).columns, trait, trait_set)
    train_scaled_weighted = scale_and_weight_train(
        df=train, fi=fi, means=train_means, stds=train_stds
    ).sample(frac=ts_cfg.train_sample, random_state=cfg.random_seed)

    log.info("Calculating average pairwise distance for training data...")
    if not syscfg[trait_set].chunked_dist:
        avg_train_dist = average_train_distance(
            train_df=train_scaled_weighted.drop(columns=["fold"]),
            batch_size=ts_cfg.avg_dist_batch_size,
            device_ids=syscfg.device_ids,
        )
    else:
        log.warning("Using chunked pairwise distance calculation for large dataset...")
        avg_train_dist = average_train_distance_chunked(
            train_df=train_scaled_weighted.drop(columns=["fold"]),
            num_chunks=40,
            device_ids=syscfg.device_ids,
        )

    log.info("Average Pairwise Distance for training data: %.4f", avg_train_dist)

    log.info("Calculating DI threshold using training data...")
    di_threshold = calc_di_threshold(
        train_scaled_weighted.sample(frac=ts_cfg.train_sample),
        avg_train_dist,
        syscfg.device_ids,
    )
    log.info("DI threshold: %.4f", di_threshold)

    log.info("Loading, scaling, and weighting predict data...")
    client, cluster = init_dask()
    pred_scaled_weighted = scale_and_weight_predict(
        df=load_predict_data(
            npartitions=ts_cfg.predict_partitions, sample=syscfg.predict_sample
        ),
        fi=fi,
        means=train_means,
        stds=train_stds,
    )
    close_dask(client, cluster)

    log.info("Computing DI values for predict data...")
    predict_di = calc_di_predict(
        predict=pred_scaled_weighted,
        train=train_scaled_weighted.drop(columns=["fold"]).sample(
            frac=ts_cfg.train_sample
        ),
        mean_distance=avg_train_dist,
        di_threshold=di_threshold,
        device_ids=syscfg.device_ids,
    )

    log.info("DI stats for %s (%s):", trait, trait_set)
    log.info(predict_di["di"].describe())

    log.info("Writing %s...", out_path)
    aoa_r = rasterize_points(predict_di, res=cfg.target_resolution, crs=cfg.crs)
    xr_to_raster(aoa_r, out_path)

    log.info("Cleaning up...")
    train_fn.unlink()
    log.info("Done! ✅")


def main(args: argparse.Namespace = cli(), cfg: ConfigBox = get_config()) -> None:
    """Main function for the AOA analysis."""

    aoa_dirs = [get_aoa_dir() / trait for trait in get_active_traits()]
    syscfg = cfg[detect_system()][cfg.model_res]["aoa"]

    for d in aoa_dirs:
        trait = d.name
        for ts in ["splot_gbif", "splot"]:
            ts_aoa_fn = d / ts / f"{trait}_{ts}_aoa.tif"
            ts_aoa_fn.parent.mkdir(parents=True, exist_ok=True)

            if ts_aoa_fn.exists() and not args.overwrite:
                log.info("Skipping existing AoA file: %s", ts_aoa_fn)
                continue

            calc_aoa(
                trait=trait,
                trait_set=ts,
                out_path=ts_aoa_fn,
                cfg=cfg,
                syscfg=syscfg,
                overwrite=args.overwrite,
            )

    log.info("Done! 🎉")


if __name__ == "__main__":
    main()

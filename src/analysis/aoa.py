"""Calculate Area of Applicability (AOA) for a given trait"""

import argparse
from pathlib import Path

import cudf
import cupy as cp
import dask.array as da
import dask.dataframe as dd
import pandas as pd
from box import ConfigBox
from cuml.metrics import pairwise_distances
from cuml.neighbors import NearestNeighbors
from dask_cuda import LocalCUDACluster
from distributed import Client, LocalCluster

from src.conf.conf import get_config
from src.conf.environment import log
from src.utils.dataset_utils import (
    get_latest_run,
    get_predict_imputed_fn,
    get_trait_models_dir,
    get_y_fn,
)

# from src.utils.log_utils import suppress_dask_logging
from src.utils.training_utils import assign_splits, filter_trait_set, set_yx_index

Y_COL: str = "X11_mean"
TRAIT_SET: str = "splot"
CFG: ConfigBox = get_config()
DEVICE_IDS = (0, 1, 2, 3)
TRAIN_SAMPLE = 1
PREDICT_SAMPLE = 0.5
PREDICT_PARTITIONS = 5


def cli() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate Area of Applicability (AOA)"
    )
    parser.add_argument(
        "--trait",
        type=str,
        default=Y_COL,
        help="Trait to calculate AOA for",
    )
    parser.add_argument(
        "--trait_set",
        type=str,
        default=TRAIT_SET,
        help="Trait set to use for training and prediction",
    )
    parser.add_argument(
        "--train_sample",
        type=float,
        default=TRAIN_SAMPLE,
        help="Fraction of training data to sample",
    )
    parser.add_argument(
        "--predict_sample",
        type=float,
        default=PREDICT_SAMPLE,
        help="Fraction of predict data to sample",
    )
    parser.add_argument(
        "--predict_partitions",
        type=int,
        default=PREDICT_PARTITIONS,
        help="Number of partitions for the predict data",
    )
    parser.add_argument("--device-ids", type=int, nargs="+", default=DEVICE_IDS)
    return parser.parse_args()


def init_dask(
    cuda: bool = False,
    device_ids: tuple[int, ...] | None = None,
    memory_limit: str = "auto",
) -> tuple[Client, LocalCluster]:
    """Initialize the Dask client and cluster."""
    if cuda:
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=device_ids, memory_limit=memory_limit
        )
    else:
        cluster = LocalCluster(
            dashboard_address=CFG.dask_dashboard, memory_limit=memory_limit
        )

    client = Client(cluster)
    return client, cluster


def close_dask(client: Client, cluster: LocalCluster | LocalCUDACluster) -> None:
    """Close the Dask client and cluster."""
    client.close()
    cluster.close()


def _to_dd(
    df: pd.DataFrame, npartitions: int
) -> dd.DataFrame:  # pyright: ignore[reportPrivateImportUsage]
    return dd.from_pandas(  # pyright: ignore[reportPrivateImportUsage]
        df, npartitions=npartitions
    )


def load_train_data(
    y_col: str, trait_set: str, sample: int | float = 1
) -> pd.DataFrame:
    """Load the training data for the AOA analysis."""
    # suppress_dask_logging()

    with Client(n_workers=20, dashboard_address=CFG.dask_dashboard):
        train = (
            pd.read_parquet(get_y_fn(), columns=["x", "y", y_col, "source"])
            # .pipe(pipe_log, "Setting yx index and assigning splits...")
            .pipe(set_yx_index)
            .pipe(assign_splits, label_col=y_col)
            .groupby("fold", group_keys=False)
            .sample(frac=sample, random_state=CFG.random_seed)
            .reset_index()
            # .pipe(pipe_log, "Filtering trait set...")
            .pipe(filter_trait_set, trait_set=trait_set)
            .drop(columns=[y_col, "source"])
            # .pipe(pipe_log, "Converting to dask dataframe...")
            .pipe(_to_dd, npartitions=50)
            # .pipe(pipe_log, "Merging with imputed predict data...")
            .merge(
                # Merge using inner join with the imputed predict data (described below)
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


def load_predict_data(npartitions: int = 50, sample: int | float = 1) -> dd.DataFrame:
    """Load the imputed predict data for the AOA analysis."""
    return dd.read_parquet(get_predict_imputed_fn(), npartitions=npartitions).sample(
        frac=sample, random_state=CFG.random_seed
    )


def _scale(df: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
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


def _weight(df: pd.DataFrame, fi: pd.DataFrame, dask: bool = False) -> pd.DataFrame:
    if dask:
        return dd.concat([df * fi.T.values], axis=1)

    return pd.concat([df * fi.T.values], axis=1)


def _scale_and_weight_train(
    df: pd.DataFrame, fi: pd.DataFrame, means: pd.Series, stds: pd.Series
) -> pd.DataFrame:
    folds = df[["fold"]].copy()
    df = df.drop(columns=["fold"])
    return pd.concat([_weight(_scale(df, means, stds), fi), folds], axis=1)


def _scale_and_weight_predict(
    df: dd.DataFrame, fi: pd.DataFrame, means: pd.Series, stds: pd.Series
) -> dd.DataFrame:
    xy = df[["x", "y"]].copy()
    df = df.drop(columns=["x", "y"])
    return dd.concat(
        [xy, _weight(df.map_partitions(_scale, means, stds), fi, dask=True)], axis=1
    )


def _df_to_cupy(df: pd.DataFrame, device_id: int) -> cp.ndarray:
    """Convert a Pandas DataFrame to CuPy array on a specific GPU device."""
    with cp.cuda.Device(device_id):
        return cudf.DataFrame.from_pandas(df).to_cupy()


def _df_to_cudf(df: pd.DataFrame, device_id: int) -> cudf.DataFrame:
    """Convert a Pandas DataFrame to CuDF DataFrame on a specific GPU device."""
    with cp.cuda.Device(device_id):
        return cudf.DataFrame.from_pandas(df)


def _cupy_to_dask(data: cp.ndarray, num_chunks: int) -> da.Array:
    """Convert a CuPy array into a Dask array with specified number of chunks."""
    return da.from_array(data, chunks=(data.shape[0] // num_chunks, data.shape[1]))


# def _batch_pairwise_distances(
#     data: cp.ndarray, start_idx: int, end_idx: int
# ) -> cp.ndarray:
#     batch_data = data[start_idx:end_idx]
#     distances = pairwise_distances(data, batch_data)
#     avg_distances = cp.mean(distances, axis=1)
#     return avg_distances


# def _batch_pairwise_distances(
#     data: da.Array, start_idx: int, end_idx: int
# ) -> cp.ndarray:
#     """Process a batch of data to compute pairwise distances on GPU."""
#     # Extract batch and compute pairwise distances
#     batch_data = data[start_idx:end_idx].compute()  # Smaller batch to worker
#     distances = pairwise_distances(
#         cp.asarray(data), cp.asarray(batch_data), metric="euclidean"
#     )
#     avg_distances = cp.mean(distances, axis=1)
#     return avg_distances


def average_train_distance_chunked(
    train_df: pd.DataFrame, num_chunks: int, device_ids: tuple[int, ...]
) -> float:
    """Compute the mean of the average pairwise distances for the training data using chunked pairwise distance calculations."""

    def _process_chunk(chunk1: cp.ndarray, chunk2: cp.ndarray) -> cp.ndarray:
        distances = pairwise_distances(chunk1, chunk2, metric="euclidean")
        avg_distances = cp.mean(distances)
        return avg_distances

    cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=device_ids)
    client = Client(cluster)

    # Convert data to CuPy for GPU processing
    cupy_data = _df_to_cupy(train_df, device_ids[0])

    # Define chunk size based on the number of chunks
    n_samples = train_df.shape[0]
    chunk_size = (n_samples + num_chunks - 1) // num_chunks

    # Divide data into chunks
    chunks = [cupy_data[i : i + chunk_size] for i in range(0, n_samples, chunk_size)]

    # Scatter each chunk separately
    scattered_chunks = client.scatter(chunks)

    # Create Dask futures for each chunk pair
    futures = []
    for i, chunk1 in enumerate(scattered_chunks):
        for j, chunk2 in enumerate(scattered_chunks):
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

    client, cluster = init_dask(cuda=True, device_ids=device_ids)

    # Convert data to CuPy for GPU processing
    cupy_data = _df_to_cupy(train_df, device_ids[0])
    scattered_data = client.scatter(cupy_data, broadcast=True)

    # Define number of batches
    n_samples = train_df.shape[0]
    num_batches = (n_samples + batch_size - 1) // batch_size

    # Create Dask delayed tasks for each batch
    futures = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        future = client.submit(_process_batch, scattered_data, start_idx, end_idx)
        futures.append(future)

    batch_results = client.gather(futures)

    # results = []
    # for future in futures:
    #     try:
    #         result = future.result()
    #         results.append(result)
    #     except Exception as e:
    #         log.error(f"Future failed with exception: {e}")

    close_dask(client, cluster)

    # Sum the results and normalize by the number of batches
    avg_distances = cp.sum(cp.array(batch_results), axis=0) / num_batches

    # Compute the overall mean of the average distances
    return cp.mean(avg_distances).item()  # Convert from CuPy to Python float


# def average_train_distance(
#     train_df: pd.DataFrame, batch_size: int, device_ids: tuple[int, ...]
# ) -> float:
#     """
#     Compute average pairwise distances using CuML's pairwise_distances, leveraging Dask and GPU.
#     :param dataframe: Dask-cuDF dataframe containing the dataset
#     :param batch_size: Number of rows in each batch/partition
#     :return: Average pairwise distance across all observations
#     """
#     cluster = LocalCUDACluster(CUDA_VISIBLE_DEVICES=device_ids)
#     client = Client(cluster)

#     n = len(train_df)

#     # Repartition the dataframe into chunks to fit in GPU memory
#     train_ddf = dask_cudf.from_cudf(_df_to_cudf(train_df, device_ids[0])).repartition(
#         npartitions=n // batch_size
#     )

#     total_sum = 0
#     total_pairs = 0

#     # Compute pairwise distances between all partition pairs
#     def chunked_distance(part1: dask_cudf.DataFrame, part2: dask_cudf.DataFrame):
#         # Compute pairwise distances using CuML's optimized function
#         dists = pairwise_distances(part1.to_cupy(), part2.to_cupy())
#         return dists.sum(), dists.size  # Return the sum and number of distances

#     results = []

#     # Loop through all partition pairs (both intra- and inter-chunk)
#     for i in range(train_ddf.npartitions):
#         df1 = train_ddf.get_partition(i)
#         for j in range(i, train_ddf.npartitions):
#             df2 = train_ddf.get_partition(j)
#             result = delayed(chunked_distance)(df1, df2)
#             results.append(result)

#     # Collect the results
#     distances_and_pairs = compute(*results)

#     close_dask(client, cluster)

#     # Sum all distances and count pairs
#     total_sum = sum(d[0] for d in distances_and_pairs)
#     total_pairs = sum(d[1] for d in distances_and_pairs)

#     avg_distance = total_sum / total_pairs

#     return avg_distance


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


def calculate_fold_min_distances(
    fold_data: cp.ndarray, other_fold_data: cp.ndarray
) -> cp.ndarray:
    nn_model = NearestNeighbors(n_neighbors=1, algorithm="brute")
    nn_model.fit(other_fold_data)
    distances, _ = nn_model.kneighbors(fold_data)
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

    client, cluster = init_dask(cuda=True, device_ids=device_ids[1:])
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
                calculate_fold_min_distances, fold_data, other_fold_data, retries=10
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
    client, cluster = init_dask(cuda=True, device_ids=device_ids)

    predict_gpu = predict.to_backend("cudf")
    train_gpu = dd.from_pandas(train).to_backend("cudf")

    def compute_nearest_neighbors(
        pred_partition: cudf.DataFrame, train_df: cudf.DataFrame
    ) -> cudf.DataFrame:
        nn_model = NearestNeighbors(n_neighbors=1, algorithm="brute")
        nn_model.fit(train_df)
        pred_partition_xy = pred_partition[["x", "y"]]
        distances, _ = nn_model.kneighbors(pred_partition.drop(columns=["x", "y"]))
        result = cudf.DataFrame(
            {
                "x": pred_partition_xy["x"],
                "y": pred_partition_xy["y"],
                "distance": distances,
            },
            index=pred_partition.index,
        )
        return result

    distances = predict_gpu.map_partitions(compute_nearest_neighbors, train_gpu)

    distances = distances.compute()

    close_dask(client, cluster)

    distances["di"] = distances["distance"] / mean_distance
    distances["aoa"] = distances["di"] > di_threshold
    return distances


def main(args: argparse.Namespace = cli()) -> None:
    """Main function for the AOA analysis."""
    train_fn = Path(f"{args.trait}_{args.trait_set}.parquet")

    if train_fn.exists():
        log.info("Loading existing training data...")
        train = pd.read_parquet(train_fn)

    else:
        log.info("Generating training data...")
        train = load_train_data(
            sample=args.train_sample, y_col=args.trait, trait_set=args.trait_set
        )
        log.info("Writing training data to disk...")
        train.to_parquet(train_fn, compression="zstd")

    log.info("Scaling and weighting training data...")
    train_means = train.drop(columns=["fold"]).mean()
    train_stds = train.drop(columns=["fold"]).std()

    fi = load_feature_importance(
        train.drop(columns=["fold"]).columns, args.trait, args.trait_set
    )
    train_scaled_weighted = _scale_and_weight_train(
        df=train, fi=fi, means=train_means, stds=train_stds
    )

    log.info("Calculating average pairwise distance for training data...")
    if args.trait_set == "splot":
        avg_train_dist = average_train_distance(
            train_df=train_scaled_weighted.drop(columns=["fold"]).sample(
                frac=args.train_sample
            ),
            batch_size=10000,
            device_ids=args.device_ids,
        )
    else:
        avg_train_dist = 0.8633
        # log.warning("Using chunked pairwise distance calculation for large dataset...")
        # avg_train_dist = average_train_distance_chunked(
        #     train_df=train_scaled_weighted.drop(columns=["fold"]).sample(
        #         frac=0.5, random_state=CFG.random_seed
        #     ),
        #     num_chunks=40,
        #     device_ids=args.device_ids,
        # )

    log.info("Average Pairwise Distance for training data: %.4f", avg_train_dist)

    # log.info(
    #     "Writing scaled and weighted training data to temporary file (addresses strange "
    #     "Dask CUDA bug)..."
    # )
    # with tempfile.TemporaryFile() as f:
    #     train_scaled_weighted.to_parquet(f, compression="zstd")
    #     train_scaled_weighted = pd.read_parquet(f)

    log.info("Calculating DI threshold using training data...")
    if args.trait_set == "splot":
        di_threshold = calc_di_threshold(
            train_scaled_weighted.sample(frac=args.train_sample),
            avg_train_dist,
            args.device_ids,
        )
    else:
        di_threshold = 0.3056
    log.info("DI threshold: %.4f", di_threshold)

    # Load, scale, and weight predict data
    log.info("Loading, scaling, and weighting predict data...")
    client, cluster = init_dask(cuda=False)

    pred_scaled_weighted = _scale_and_weight_predict(
        df=load_predict_data(
            npartitions=args.predict_partitions, sample=args.predict_sample
        ),
        fi=fi,
        means=train_means,
        stds=train_stds,
    )

    close_dask(client, cluster)

    # Compute DI for predict data
    log.info("Computing DI values for predict data...")
    predict_di = calc_di_predict(
        predict=pred_scaled_weighted,
        train=train_scaled_weighted.drop(columns=["fold"]).sample(
            frac=args.train_sample
        ),
        mean_distance=avg_train_dist,
        di_threshold=di_threshold,
        device_ids=args.device_ids,
        # num_chunks_predict=args.predict_partitions,
        # num_chunks_train=20,
    )

    log.info("Saving DI and AoA for predict data...")
    predict_di.to_parquet(
        f"{args.trait}_DI_{args.trait_set}.parquet", compression="zstd"
    )


if __name__ == "__main__":
    main()

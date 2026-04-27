import argparse
import pickle
import sys
import tempfile
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmarks.core.str_utils import compute_str_ranges, to_str_format
from str.model.STRmodel import ExpSTRmodel
from utils.traj_distance import MEASURE_FUNCS, calculate_distance

warnings.filterwarnings("ignore")

# Worker-local trajectory array, populated once per worker process via initializer.
_worker_arrs: list[np.ndarray] | None = None


def _init_worker(arrs: list[np.ndarray]) -> None:
    global _worker_arrs
    _worker_arrs = arrs


def _compute_row(args: tuple[int, str]) -> tuple[int, np.ndarray]:
    """Compute distances from trajectory i to all j > i (upper triangle only)."""
    i, metric = args
    assert _worker_arrs is not None
    n = len(_worker_arrs)
    row = np.zeros(n, dtype=np.float32)
    for j in range(i + 1, n):
        row[j] = calculate_distance(metric, _worker_arrs[i], _worker_arrs[j])
    return i, row


def _compute_pairwise_matrix(
    trajs: list[list[list[float]]], metric: str, n_workers: int
) -> np.ndarray:
    """Compute the full symmetric pairwise distance matrix in parallel."""
    n = len(trajs)
    arrs = [np.array(t, dtype=np.float64) for t in trajs]
    matrix = np.zeros((n, n), dtype=np.float32)
    tasks = [(i, metric) for i in range(n)]

    with Pool(n_workers, initializer=_init_worker, initargs=(arrs,)) as pool:
        for i, row in tqdm(
            pool.imap_unordered(_compute_row, tasks),
            total=n,
            desc=f"Computing {metric} matrix ({n_workers} workers)",
        ):
            matrix[i] = row

    # Upper triangle was filled; mirror to lower triangle.
    return matrix + matrix.T


def prepare_data(
    data_path: Path,
    simi_metric: str,
    n_workers: int,
    tmp_dir: Path,
) -> tuple[Path, Path, int, dict]:
    """Load parquet, convert to STR format, compute similarity matrix.

    Writes pkl files to tmp_dir (one per run) so concurrent HPC jobs don't
    collide. Returns (traj_pkl, matrix_pkl, n_trajs, range_config).
    """
    df = pl.read_parquet(data_path)
    trajs_xy: list[list[list[float]]] = df["TRAJ_MERCATOR"].to_list()
    timestamps = df["TIMESTAMPS"].to_list() if "TIMESTAMPS" in df.columns else None
    str_trajs = to_str_format(trajs_xy, timestamps=timestamps)
    n_trajs = len(str_trajs)

    traj_pkl = tmp_dir / "str_trajs.pkl"
    matrix_pkl = tmp_dir / f"{simi_metric}_matrix.pkl"

    with open(traj_pkl, "wb") as f:
        pickle.dump(str_trajs, f)

    matrix = _compute_pairwise_matrix(trajs_xy, simi_metric, n_workers)
    with open(matrix_pkl, "wb") as f:
        pickle.dump(matrix, f)

    x_range, y_range, z_range = compute_str_ranges(str_trajs)
    data_features = [x_range, y_range, z_range, [0, 7], [0, 480], [0, 7], [0, 480]]
    range_config = {
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "data_features": data_features,
    }

    return traj_pkl, matrix_pkl, n_trajs, range_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STR training")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the preprocessed parquet file.",
    )
    parser.add_argument(
        "--simi-metric",
        type=str,
        default="dtw",
        choices=sorted(MEASURE_FUNCS),
        help="Trajectory similarity metric used to build the training matrix.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "config.yaml",
        help="Path to the STR config yaml.",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument(
        "--just-embedding",
        action="store_true",
        help="Skip training and only compute embeddings.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel workers for similarity matrix computation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    _ckpt_dir = PROJECT_ROOT / "models" / "str"
    _ckpt_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as _tmp:
        tmp_dir = Path(_tmp)
        traj_pkl, matrix_pkl, n_trajs, range_config = prepare_data(
            args.data_path, args.simi_metric, args.workers, tmp_dir
        )

        config["data"] = args.data_path.stem
        config["dis_type"] = args.simi_metric
        config["traj_num"] = n_trajs
        config["traj_path"] = str(traj_pkl)
        config["stdis_matrix_path"] = str(matrix_pkl)
        config.update(range_config)

        train_end = int(n_trajs * 0.8)
        config["train_data_range"] = [0, train_end]
        config["val_data_range"] = [train_end, n_trajs]

        config["model_best_wts_path"] = str(
            _ckpt_dir / f"{config['data']}_{config['model']}_{config['dis_type']}_best.pt"
        )
        config["model_best_topAcc_path"] = str(
            _ckpt_dir / f"{config['data']}_{config['model']}_{config['dis_type']}_topAcc.csv"
        )
        config["embeddings_path"] = str(
            _ckpt_dir / f"{config['data']}_{config['model']}_{config['dis_type']}_embeddings.pkl"
        )

        print("Config:", config)

        exp = ExpSTRmodel(
            config=config,
            gpu_id=args.gpu,
            load_model=args.load_model,
            just_embeddings=args.just_embedding,
        )
        if args.just_embedding:
            exp.embedding()
        else:
            exp.train()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

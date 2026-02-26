"""Command-line interface for gaur-sql-detect."""

import os

# Force device selection before any torch imports.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")

import argparse
import logging
import numpy as np
import pandas as pd
import random
import sys
from logging.handlers import TimedRotatingFileHandler
from sklearn.model_selection import train_test_split

import gaur_sqld
from gaur_sqld.utils.trainers import train_models

logger = logging.getLogger(__name__)


def init_logging(args: argparse.Namespace) -> None:
    lf = TimedRotatingFileHandler(
        gaur_sqld.ppths.logs_path + "/training.log",
        when="midnight",
    )

    lg_lvl = logging.DEBUG if args.debug else logging.INFO
    lf.setLevel(lg_lvl)
    lstdo = logging.StreamHandler(sys.stdout)
    lstdo.setLevel(lg_lvl)

    lstdof = logging.Formatter(" %(message)s")
    lstdo.setFormatter(lstdof)
    logging.basicConfig(level=lg_lvl, handlers=[lf, lstdo])


def init_args() -> argparse.Namespace:
    """Argparse initializing function."""
    parser = argparse.ArgumentParser(
        prog="gaur-sql-detect",
        description="Train and evaluate GAUR-based IDS on the SQL Injection Detection task.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Prints more details about model training",
    )

    parser.add_argument(
        "--fixed-fpr",
        type=float,
        default=None,
        dest="fixed_fpr",
        help="Override saved threshold by computing one from test set normal samples "
        "at the given FPR (e.g. 0.01 for 1%% FPR)",
    )

    parser.add_argument(
        "--testing",
        action="store_true",
        help="Reduce dataset size to test correct code execution",
    )

    parser.add_argument(
        "--use-datadir",
        action="store_true",
        dest="use_datadir",
        help="Use the cache_datadir_path instead of cache_path for the cache mechanism.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        dest="dataset",
        default="./data/dataset.csv",
        help="Filepath to the dataset. Defaults to './data/dataset.csv'.",
    )

    parser.add_argument(
        "--subfolder",
        dest="subfolder",
        help="Save results in output subfolder. Used when computing on multiple nodes "
        "to prevent results overwrite.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to train (e.g., --models ocsvm ae). Use 'all' to run everything.",
    )

    parser.add_argument(
        "--trace-type",
        nargs="+",
        dest="trace_type",
        default=["expert"],
        help="Define which trace format to expect to collect. Valid values are: "
        "all, expert, chatgpt, claude, gpt-oss, llama, mistral.",
    )

    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Do not use traces cache for GAUR-based pipelines.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a TOML configuration file.",
    )

    return parser.parse_args()


def main() -> None:
    args = init_args()

    if args.config:
        gaur_sqld.configure_from_file(args.config)

    import gaur_sqld.config as _cfg

    seed = _cfg.seed
    np.random.seed(seed)
    random.seed(seed)

    init_logging(args)

    df = pd.read_csv(
        args.dataset,
        dtype={
            "full_query": str,
            "label": int,
            "user_inputs": str,
            "attack_stage": str,
            "tamper_method": str,
            "attack_status": str,
            "statement_type": str,
            "query_template_id": str,
            "attack_id": str,
            "attack_technique": str,
            "split": str,
        },
    )

    if args.testing:
        df = df.sample(n=500, random_state=seed)

    if args.subfolder:
        gaur_sqld.ppths.set_output_subfolder(args.subfolder)

    # Compute validation split now.
    _train_idx = df[df["split"] == "train"].index
    train_idx, val_idx = train_test_split(
        _train_idx,
        test_size=0.1,
        random_state=seed,
    )
    df.loc[val_idx, "split"] = "val"

    train_models(df, args)


if __name__ == "__main__":
    main()

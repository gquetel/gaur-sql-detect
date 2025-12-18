"""
Script to train and evaluate GAUR-based IDS on the SQL Injection Detection task.
"""

import os

import xpgaur

# We force device on which training happens.
# device = torch.device("cuda:0" if USE_CUDA else "cpu") is not taken
# into account apparently...
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

from logging.handlers import TimedRotatingFileHandler
from sklearn.model_selection import train_test_split

import argparse
import logging
import numpy as np
import pandas as pd
import random
import sys

from xpgaur.utils.trainers import train_models

logger = logging.getLogger(__name__)

def init_logging(args):
    lf = TimedRotatingFileHandler(
        xpgaur.ppths.logs_path + "/training.log",
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
    """Argsparse initializing function.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Prints more details on about model training",
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
        help="Use /data/gquetel/ folder for cache mechasnim.",
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
        help="Save results in output subfolder. Used when computing on multiple nodes to prevent results overwrite.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to train (e.g., --models ocsvm_li ae_cv). Use 'all' to run everything.",
    )

    parser.add_argument(
        "--trace-type",
        nargs="+",
        dest="trace_type",
        default=["expert"],
        help="Define which trace format to expect to collect. Valid values are: all, expert, chatgpt, claude, gpt-oss, llama, mistral.",
    )

    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Do not use traces cache for GAUR-based pipelines.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    seed = xpgaur.seed
    np.random.seed(seed)
    random.seed(seed)
    args = init_args()
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
        # Save results in a subfolder of output.
        xpgaur.ppths.set_output_subfolder(args.subfolder)

    # Compute validation split now.
    _train_idx = df[df["split"] == "train"].index
    train_idx, val_idx = train_test_split(
        _train_idx,
        test_size=0.1,
        random_state=seed,
    )
    df.loc[val_idx, "split"] = "val"

    # Then train models.
    train_models(df, args)

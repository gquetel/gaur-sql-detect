"""
Script to sample the Anubis Dataset according to a percentage given as argument.
"""

import pandas as pd
import argparse
import os

def init_args() -> argparse.Namespace:
    """Argsparse initializing function.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        dest="dataset",
        required=True,
        help="Filepath to the dataset.",
    )

    parser.add_argument(
        "--percent",
        type=int,
        dest="percent",
        default=10,
        help=" Proportion of the dataset to keep",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = init_args()
    df = pd.read_csv(args.dataset)

    frac = args.percent / 100
    sampled_df = df.sample(frac=frac, random_state=2)

    dataset_name = os.path.basename(args.dataset)
    output_filepath = f"{args.percent}-{dataset_name}"
    sampled_df.to_csv(output_filepath, index=False)

    print(f"Sampled {args.percent}% of {args.dataset} -> {output_filepath}")

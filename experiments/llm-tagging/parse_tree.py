"""
Script to compute the parse tree of some queries that is human readable.
- First we load the traces, select some queries.
- We replace the node number by their names so that the LLM can understand it.
- We produce a "human readable" version of the tree.
"""

# Make utils & config import work.
import sys

sys.path.append("../..")

import re
import numpy as np
import pandas as pd
import random
import argparse

import gaur_sqld
from gaur_sqld.utils.traces_collector import get_traces_from_df


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

    return parser.parse_args()


def get_kind_names() -> list:
    """Return a mapping: symbkindnumber -> name.

    Returns:
        list: List of rule names.
    """
    symbkind_path = "./data/yysymbkind.c"

    with open(symbkind_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r"\s*(YYSYMBOL_\w+)\s*=\s*([-]?\d+)\s*,")
    symbols = {}
    for name, value in pattern.findall(content):
        stripped = name.replace("YYSYMBOL_", "")
        symbols[int(value)] = stripped
    
    return symbols


def get_parse_tree(df: pd.DataFrame, idx : list ):
    # rule_id is the 3 entry in the node.
    df= df.iloc[idx]
    df_traces = get_traces_from_df(df)
    df_merged = pd.concat([df_traces, df], axis=1, join="inner")

    kind_to_name = get_kind_names()
    for i,row in df_merged.iterrows():
        # print(row["full_query"])
        print_parse_tree(row["semantic_tree"], kind_to_name)


def print_parse_tree(trace: str, kind_to_name: dict):
    nodes, edges = trace.split("||-||")
    nodes = nodes.split("|")
    order_to_name = {}
    for node in nodes:
        if not node:
            continue
        r_order, yykind, _, _, _, _ = node.split(":")
        yykind = int(yykind)
        if yykind > 832:
            order_to_name[int(r_order)] = kind_to_name[yykind]
        else:
            # If token, do not display its name, we don't give tag to them anyway
                        order_to_name[int(r_order)] = kind_to_name[yykind]

            # order_to_name[int(r_order)] = str(yykind)
    edges = edges.split()
    graph = {}

    for segment in edges:
        parent, children = segment.split(">")
        parent = int(parent)
        children = [int(c) for c in children.split(":") if c]
        graph[parent] = graph.get(parent, []) + children

    # Build DOT text with mapped names
    print("digraph query {")
    for parent, children in graph.items():
        parent_name = order_to_name[parent]

        for child in children:
            child_name = order_to_name[child]
            print(f'    "{parent_name}" -> "{child_name}";')
    print("}\n")


if __name__ == "__main__":
    seed = gaur_sqld.seed
    np.random.seed(seed)
    random.seed(seed)

    args = init_args()

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
    get_parse_tree(df, [])

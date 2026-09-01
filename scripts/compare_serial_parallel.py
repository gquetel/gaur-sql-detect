#!/usr/bin/env python3
"""Compare serial vs parallel GAUR trace collection.

Collects traces for N test-split rows from each in-domain superviz26-lodo
split (a-a, b-b, c-c, d-d) twice: once with n_workers=1 (serial) and once
with n_workers=N_WORKERS_PARALLEL, each against a freshly (re)started
gaur-expert server. Reports whether the two collections produced identical
traces and how long each took.

Usage:
    python3 scripts/compare_serial_parallel.py [n_rows] [n_workers_parallel]

Defaults to 100000 rows and 32 workers.
"""
import os
import sys
import time

import pandas as pd

from gaur_sqld import config as cfg
from gaur_sqld.server import ensure_server, stop_server
from gaur_sqld.utils.traces_collector import get_traces_from_df

DATASETS = ["a-a", "b-b", "c-c", "d-d"]
DATA_DIR = os.path.expanduser("~/datasets/superviz26-lodo")
COMPARE_COLUMNS = [
    # query_id is a per-connection parser-invocation counter; serial uses one
    # connection and parallel uses one per worker, so it is expected to
    # differ and says nothing about trace correctness.
    "n_terminal",
    "n_nonterminal",
    "is_syntax_error",
    "semantic_tree",
    "depth",
    "n_parser_invoc",
]


def load_test_rows(name: str, n: int) -> pd.DataFrame:
    path = f"{DATA_DIR}/{name}.csv"
    chunks = []
    got = 0
    for chunk in pd.read_csv(path, chunksize=200_000):
        test_chunk = chunk[chunk["split"] == "test"]
        chunks.append(test_chunk)
        got += len(test_chunk)
        if got >= n:
            break
    df = pd.concat(chunks, ignore_index=True)
    df = df[df["split"] == "test"].head(n).reset_index(drop=True)
    if len(df) < n:
        print(f"WARNING: {name} only has {len(df)} test rows (< {n})")
    return df


def run(df: pd.DataFrame, n_workers: int, label: str) -> tuple[pd.DataFrame, float]:
    stop_server(cfg.trace_type)
    # Start the server outside the timed region: get_traces_from_df() calls
    # ensure_server() itself, and the Nix script reinitialises the datadir,
    # which would otherwise be counted as collection time.
    ensure_server(cfg.trace_type)
    t0 = time.time()
    traces = get_traces_from_df(df, use_cache=False, disable_tqdm=False, n_workers=n_workers)
    dt = time.time() - t0
    print(f"[{label}] n_workers={n_workers} rows={len(df)} time={dt:.1f}s", flush=True)
    return traces, dt


def compare(df_serial: pd.DataFrame, df_parallel: pd.DataFrame, name: str) -> None:
    if len(df_serial) != len(df_parallel):
        print(
            f"[{name}] MISMATCH: row counts differ "
            f"(serial={len(df_serial)}, parallel={len(df_parallel)})"
        )
        return
    a = df_serial.reset_index(drop=True)
    b = df_parallel.reset_index(drop=True)
    mismatches = 0
    for col in COMPARE_COLUMNS:
        if col not in a.columns or col not in b.columns:
            continue
        diff = a[col].fillna("<NA>").astype(str) != b[col].fillna("<NA>").astype(str)
        n_diff = int(diff.sum())
        if n_diff:
            mismatches += n_diff
            print(f"[{name}] column '{col}': {n_diff} mismatched rows")
    if mismatches == 0:
        print(f"[{name}] OK: serial and parallel traces are identical ({len(a)} rows)")


def main() -> int:
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    n_workers_parallel = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    results = {}
    for name in DATASETS:
        print(f"\n=== {name} (n_rows={n_rows}) ===", flush=True)
        df = load_test_rows(name, n_rows)

        serial_traces, t_serial = run(df, 1, "serial")
        parallel_traces, t_parallel = run(df, n_workers_parallel, "parallel")

        compare(serial_traces, parallel_traces, name)
        speedup = t_serial / t_parallel if t_parallel else float("nan")
        print(f"[{name}] speedup: {speedup:.2f}x (serial={t_serial:.1f}s, parallel={t_parallel:.1f}s)")
        results[name] = {"t_serial": t_serial, "t_parallel": t_parallel, "speedup": speedup}

    print("\n=== SUMMARY ===")
    for name, r in results.items():
        print(
            f"{name}: serial={r['t_serial']:.1f}s parallel={r['t_parallel']:.1f}s "
            f"speedup={r['speedup']:.2f}x"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

from subprocess import PIPE, STDOUT, Popen
from scipy import stats

import argparse
import mysql
import mysql.connector
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import signal
import statistics


def init_args() -> argparse.Namespace:
    """Argsparse initializing function.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csckt",
        type=str,
        dest="socket_custom",
        required=True,
        help="Filepath to the socket of a custom MySQL.",
    )

    parser.add_argument(
        "--nsckt",
        type=str,
        dest="socket_normal",
        required=True,
        help="Filepath to the socket of a normal MySQL.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        dest="dataset",
        required=True,
        help="Filepath to the dataset.",
    )

    parser.add_argument(
        "--testing",
        action="store_true",
        help="Reduce dataset size to test correct code execution",
    )

    return parser.parse_args()


confidence_level = 0.95

def get_ci(series):
    ci = stats.t.interval(
        confidence_level,
        df=len(series) - 1,
        loc=np.mean(series),
        scale=np.std(series, ddof=1) / np.sqrt(len(series)),
    )
    return ci


def run_pt_kill(socket):
    user = "root"
    password = "root"
    host = "localhost"
    database = "dataset"
    cmd = (
        f"pt-kill --kill-query --user={user} --password={password} "
        f"--host={host} --socket={socket} "
        f"--database {database} "
        f"--busy-time 5 --print --run-time 1s"
    )

    print(f"Running pt-kill due to timeout: {cmd}")

    proc = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT, text=True)

    for line in proc.stdout:
        print(f"\t[pt-kill] {line.rstrip()}")

    proc.wait()


# --- Timeout handler ---
class QueryTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise QueryTimeout("Query timed out")


# --- Execute a query with a timeout ---
def execute_with_timeout(cursor, query, timeout_sec):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        cursor.execute(query)
    except QueryTimeout:
        raise QueryTimeout(f"Execution timeout after {timeout_sec}s")
    finally:
        signal.alarm(0)


def benchmark_query_single(cnxn, cnxc, query, timeout=10):
    cnxn.reconnect()
    cnxc.reconnect()

    with cnxn.cursor(buffered=True) as curn:
        # ---- Normal Server timing ----
        start = time.perf_counter()
        try:
            execute_with_timeout(curn, query, timeout)
            timen = time.perf_counter() - start
        except QueryTimeout:
            timen = None
            run_pt_kill(socket=args.socket_normal)
        except mysql.connector.Error:
            timen = time.perf_counter() - start

    with cnxc.cursor(buffered=True) as curc:
        # ---- Instrumented Server timing ----
        start = time.perf_counter()
        try:
            execute_with_timeout(curc, query, timeout)
            timec = time.perf_counter() - start
        except QueryTimeout:
            timec = None
            run_pt_kill(socket=args.socket_custom)
        except mysql.connector.Error:
            timec = time.perf_counter() - start

    return timen, timec


def compute_metrics(results, suffix: str = ""):
    # Convert results to DataFrame
    out_df = pd.DataFrame(results)

    timen_mean = out_df["timen"].mean()
    timec_mean = out_df["timec"].mean()
    overhead_mean = out_df["overhead"].mean()

    overhead_ci = get_ci(out_df["overhead"])
    pstdev = statistics.pstdev(out_df["overhead"])
    maxv = max(out_df["overhead"])

    print("Normal connection:")
    print(f"    Mean time: {timen_mean:.6f}s")

    print("Custom connection:")
    print(f"    Mean time: {timec_mean:.6f}s")

    print("Overhead:")
    print(f"    Mean time: {overhead_mean:.6f}s")
    print(f"    95% CI: [{overhead_ci}]")
    print(f"    Standard deviation: {pstdev}")
    print(f"    Max value: {maxv}")

    cutoff = df["overhead"].quantile(0.99)
    print("1% highest overhead starts at:", cutoff)


    # Save to CSV
    output_file = f"overhead_stats{suffix}.csv"
    out_df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    args = init_args()
    # Run query using non privileged user.
    user = "toto"
    password = "toto"
    seed = 2

    cnx_normal = mysql.connector.connect(
        user=user,
        password=password,
        unix_socket=args.socket_normal,
        database="dataset",
        read_timeout=10,
        connection_timeout=10,
        write_timeout=10,
    )

    cnx_custom = mysql.connector.connect(
        user=user,
        password=password,
        unix_socket=args.socket_custom,
        database="dataset",
        read_timeout=10,
        connection_timeout=10,
        write_timeout=10,
    )

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

    df = df.sample(frac=1) # Shuffle dataset
    results = []
    queries = df["full_query"].tolist()
    save_interval = max(1, len(queries) // 20)

    for idx, q in enumerate(tqdm(queries), start=1):
        try:
            timen, timec = benchmark_query_single(cnx_normal, cnx_custom, q)

            if timec and timen:
                result = {
                    "query": q,
                    "timen": timen,
                    "timec": timec,
                    "overhead": timec - timen,
                }
                results.append(result)
            if idx % save_interval == 0:
                print(f"--- Saving checkpoint ---")
                compute_metrics(results)
        
        except mysql.connector.errors.InternalError as e:
            # Unread result found, we reset the connections.
            cnx_normal = mysql.connector.connect(
                user=user,
                password=password,
                unix_socket=args.socket_normal,
                database="dataset",
                read_timeout=10,
                connection_timeout=10,
                write_timeout=10,
            )

            cnx_custom = mysql.connector.connect(
                user=user,
                password=password,
                unix_socket=args.socket_custom,
                database="dataset",
                read_timeout=10,
                connection_timeout=10,
                write_timeout=10,
            )

    # Close connections
    cnx_normal.close()
    cnx_custom.close()

    compute_metrics(results, "_final")
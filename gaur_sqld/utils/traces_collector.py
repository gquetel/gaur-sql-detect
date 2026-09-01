import csv
import hashlib
import multiprocessing
import multiprocessing.util
import os
import signal
import sys

import mysql
import gaur_sqld.config as config
import logging
import numpy as np
import pandas as pd
import mysql.connector
from tqdm import tqdm

from gaur_sqld.config import DEFAULT_N_WORKERS
from .mysql_wrapper import SQLConnector
from gaur_sqld.server import ensure_server, GaurServerError

logger = logging.getLogger(__name__)


LOG_COLUMNS = [
    "query_id",
    "n_terminal",
    "n_nonterminal",
    "is_syntax_error",
    "semantic_tree",
    "depth",
]
LOG_HEADER = ",".join(LOG_COLUMNS) + "\n"

# Integer columns; failed tree fields become NaN.
LOG_NUMERIC_COLUMNS = [
    "query_id",
    "n_terminal",
    "n_nonterminal",
    "is_syntax_error",
    "depth",
]

# Large serialized trees can exceed csv's default field limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Per-connection log files.
MODE_PER_CONNECTION = "per-connection"
# Legacy servers use one shared log file.
MODE_LEGACY = "legacy"

# Keep short runs serial to avoid pool setup overhead.
DEFAULT_PARALLEL_MIN_ROWS = 5000

# Use several chunks per worker for better load balancing.
CHUNKS_PER_WORKER = 4


def _rows_to_frame(rows: list, header: list) -> pd.DataFrame:
    """Build a trace frame from CSV rows, matching ``pd.read_csv`` dtypes."""
    df = pd.DataFrame(rows, columns=header)

    for column in LOG_NUMERIC_COLUMNS:
        if column in df.columns:
            # Match read_csv: empty numeric fields become NaN.
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "semantic_tree" in df.columns:
        df["semantic_tree"] = df["semantic_tree"].replace("", np.nan)

    return df


class GaurTraceCollector:
    """Read the GAUR trace log for one MySQL connection."""

    def __init__(self, fp_datadir: str, connection_id: int | None = None):
        self.fp_datadir = fp_datadir
        self.connection_id = None
        self.bind(connection_id)

    def bind(self, connection_id: int | None) -> None:
        """Point the collector at a connection's log file.

        ``None`` selects the shared legacy log.
        """
        if connection_id != self.connection_id:
            # Remove the old connection's file.
            self.cleanup()

        self.connection_id = connection_id
        name = "gaur.log" if connection_id is None else f"gaur.{connection_id}.log"
        self.fp_log = os.path.join(self.fp_datadir, name)

    def reset_logfile(self) -> None:
        """Empty the Gaur log file."""
        with open(self.fp_log, "w") as file:
            file.write(LOG_HEADER)

    def collect_logfile(self) -> pd.DataFrame:
        """Read traces with ``csv.reader`` and reset the log file."""
        with open(self.fp_log, newline="") as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration:
                header = list(LOG_COLUMNS)
            rows = [row for row in reader if row]

        self.reset_logfile()
        return _rows_to_frame(rows, header)

    def cleanup(self) -> None:
        """Remove this connection's log file."""
        if self.connection_id is None:
            return
        try:
            os.remove(self.fp_log)
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"Could not remove {self.fp_log}: {e}")


def detect_log_mode(fp_datadir: str, sqlc: SQLConnector) -> str:
    """Detect whether the server uses per-connection or legacy logs."""
    if sqlc.cnx is None or not sqlc.cnx.is_connected():
        sqlc.init_new_cnx()

    cid = sqlc.cnx.connection_id
    # Trigger log creation before checking the filename.
    sqlc.execute_query("SELECT 1")

    fp_probe = os.path.join(fp_datadir, f"gaur.{cid}.log")
    if os.path.isfile(fp_probe):
        # The probe file is not needed after detection.
        try:
            os.remove(fp_probe)
        except OSError as e:
            logger.warning(f"Could not remove probe log {fp_probe}: {e}")
        return MODE_PER_CONNECTION

    if os.path.isfile(os.path.join(fp_datadir, "gaur.log")):
        logger.warning(
            "Instrumented server writes a single shared gaur.log. Parallel "
            "collection is unavailable; rebuild the server from "
            "gaur-instrumented-apps to enable it."
        )
        return MODE_LEGACY

    raise GaurServerError(
        f"No GAUR trace log found in {fp_datadir} after probing the server. "
        "Is this server actually instrumented with GAUR?"
    )


def merge_traces(df_traces: pd.DataFrame, query: str) -> pd.DataFrame:
    """Merge traces from multiple parser invocations."""

    # query_id is for tracking, so keep the first value.
    qid = df_traces.iloc[0]["query_id"]
    n_terminal = df_traces["n_terminal"].sum()
    n_nonterminal = df_traces["n_nonterminal"].sum()
    depth = df_traces["depth"].sum()
    n_parser_invoc = df_traces.shape[0]
    is_syntax_error = (df_traces["is_syntax_error"] == 1).any()

    nodes = []
    edges = []

    for tree in df_traces["semantic_tree"]:
        # Each trace contains nodes and edges separated by "||-||".
        if isinstance(tree, str):
            parts = tree.split("||-||")
            if len(parts) == 2:
                n, e = parts
                nodes.append(n)
                edges.append(e)
            else:
                logger.warning(f"Unexpected split output for query {query}: {parts}")
        else:
            # Ignore empty traces.
            logger.warning(f"A trace is invalid for query: {query}")

    # Join all node and edge fragments.
    str_node = "".join(nodes)
    str_edges = "".join(edges)

    semantic_tree = str_node + "||-||" + str_edges

    _df = pd.DataFrame(
        {
            "query_id": [qid],
            "n_terminal": n_terminal,
            "n_nonterminal": n_nonterminal,
            "is_syntax_error": is_syntax_error,
            "semantic_tree": semantic_tree,
            "depth": depth,
            "n_parser_invoc": int(n_parser_invoc),
        },
        index=[0],
    )
    return _df


def get_traces_from_query(
    query: str, sqlc: SQLConnector, gtc: GaurTraceCollector
) -> pd.DataFrame:
    """Execute a query and return its GAUR trace."""

    if sqlc.cnx is None or not sqlc.cnx.is_connected():
        # Connection setup creates parser invocations; clear them below.
        sqlc.init_new_cnx()
        # Rebind after reconnecting because the connection id changes.
        if gtc.connection_id is not None:
            gtc.bind(sqlc.cnx.connection_id)
        gtc.reset_logfile()

    # TODO: Handle queries that hang without risking a server crash.
    retcode = sqlc.execute_query(query)

    if retcode == 1:
        gtc.reset_logfile()

        # Keep the row so callers can remove it after collection.
        return pd.DataFrame(
            {
                "query_id": pd.NA,
                "n_terminal": pd.NA,
                "n_nonterminal": pd.NA,
                "is_syntax_error": pd.NA,
                "semantic_tree": pd.NA,
                "depth": pd.NA,
                "n_parser_invoc": pd.NA,
            },
            index=[0],
        )

    df_traces = gtc.collect_logfile()

    # Merge traces from stacked queries.
    if df_traces.shape[0] > 1:
        return merge_traces(df_traces=df_traces, query=query)

    df_traces["n_parser_invoc"] = 1
    return df_traces


def _partial_path(base_dir: str, str_hash_df: str, chunk_id: int, n_chunks: int) -> str:
    """Return the partial-results path for a chunk."""
    if n_chunks == 1:
        return f"{base_dir}{str_hash_df}-partial.pkl"
    return f"{base_dir}{str_hash_df}-partial-w{chunk_id}of{n_chunks}.pkl"


# Reuse one connection per worker across chunks.
_worker_conn_state: tuple | None = None


def _worker_conn(log_mode: str) -> tuple:
    """Return this process's connector and trace collector."""
    global _worker_conn_state

    if _worker_conn_state is not None:
        sqlc, gtc = _worker_conn_state
        if sqlc.cnx is not None and sqlc.cnx.is_connected():
            return sqlc, gtc
        # Do not reuse a collector whose connection has died.
        _close_worker_conn()

    sqlc = SQLConnector(
        user=config.mysql_info.user,
        pwd=config.mysql_info.password,
        socket_path=config.mysql_info.socket_path,
        database=config.mysql_info.database,
    )
    gtc = GaurTraceCollector(fp_datadir=config.mysql_info.datadir_path)

    # Bind before queries run; reconnects are handled by get_traces_from_query.
    sqlc.init_new_cnx()
    if log_mode == MODE_PER_CONNECTION:
        gtc.bind(sqlc.cnx.connection_id)
    gtc.reset_logfile()

    _worker_conn_state = (sqlc, gtc)
    return sqlc, gtc


def _close_worker_conn() -> None:
    """Close this process's connection and remove its trace log."""
    global _worker_conn_state

    if _worker_conn_state is None:
        return
    sqlc, gtc = _worker_conn_state
    _worker_conn_state = None

    # Close first so the server cannot recreate the file.
    try:
        if sqlc.cnx is not None and sqlc.cnx.is_connected():
            sqlc.cnx.close()
    except mysql.connector.Error as e:
        logger.warning(f"Could not close connection cleanly: {e}")
    gtc.cleanup()


def _worker_init() -> None:
    """Set up cleanup handlers for a pool worker."""
    # Do not inherit the parent's connection state.
    global _worker_conn_state
    _worker_conn_state = None

    multiprocessing.util.Finalize(None, _close_worker_conn, exitpriority=16)

    def on_sigterm(signum, frame):
        _close_worker_conn()
        # Avoid running interpreter shutdown from the signal handler.
        os._exit(0)

    signal.signal(signal.SIGTERM, on_sigterm)


def _collect_chunk(
    chunk: pd.DataFrame,
    fp_partial: str,
    use_cache: bool,
    log_mode: str,
    on_row=None,
) -> pd.DataFrame:
    """Collect traces for one input chunk."""
    ltraces = []
    start_idx = 0

    if use_cache and os.path.isfile(fp_partial):
        partial_df = pd.read_pickle(fp_partial, compression="zstd")
        ltraces.append(partial_df)
        start_idx = len(partial_df)
        logger.info(f"Resuming from row {start_idx}/{len(chunk)} of {fp_partial}")

    sqlc, gtc = _worker_conn(log_mode)

    save_interval = max(1, len(chunk) // 10)

    def checkpoint(reason: str) -> None:
        if not (use_cache and ltraces):
            return
        partial_df = pd.concat(ltraces)
        partial_df.index = chunk.index[: len(partial_df)]
        pd.to_pickle(partial_df, fp_partial, compression="zstd")
        logger.info(f"{reason} saved to {fp_partial}")

    try:
        for i, row in enumerate(chunk.itertuples(index=False)):
            if i < start_idx:
                if on_row is not None:
                    on_row()
                continue
            try:
                ltraces.append(get_traces_from_query(row.full_query, sqlc, gtc))
            except mysql.connector.errors.InterfaceError as e:
                logger.error(f"MySQL interface error on row {i}: {e}")
                # Preserve completed rows before propagating the error.
                checkpoint("Partial results")
                raise
            finally:
                if on_row is not None:
                    on_row()

            if use_cache and i % save_interval == 0:
                checkpoint(f"Progress checkpoint at row {i + 1}/{len(chunk)}")
    except BaseException:
        # Drop the chunk's connection after a failure.
        _close_worker_conn()
        raise

    if not ltraces:
        return pd.DataFrame()

    # Return positional rows; input indexes may contain duplicates.
    return pd.concat(ltraces, ignore_index=True)


def _collect_chunk_task(args: tuple) -> tuple:
    """Pool entry point for collecting one chunk."""
    chunk_id, chunk, fp_partial, use_cache, log_mode = args
    return chunk_id, _collect_chunk(chunk, fp_partial, use_cache, log_mode)


def _resolve_n_workers(n_workers: int, n_rows: int) -> int:
    """Resolve the requested worker count."""
    if n_workers == DEFAULT_N_WORKERS:
        if n_rows <= DEFAULT_PARALLEL_MIN_ROWS:
            return 1
        return os.cpu_count() or 1

    if n_workers < 1:
        raise ValueError(
            f"n_workers must be >= 1 or {DEFAULT_N_WORKERS} (auto), "
            f"got {n_workers}"
        )
    return n_workers


def get_traces_from_df(
    df: pd.DataFrame,
    use_cache: bool = True,
    disable_tqdm: bool = False,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Return GAUR traces for a DataFrame, using the cache when enabled."""
    if n_workers is None:
        n_workers = getattr(config, "n_workers", DEFAULT_N_WORKERS)
    n_workers = _resolve_n_workers(n_workers, len(df))

    # Use a content hash to locate cached traces.
    str_hash_df = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

    cache_dir = config.ppths.cache_path

    if use_cache:
        fp_cache = f"{cache_dir}{str_hash_df}.pkl"
        if os.path.isfile(fp_cache):
            logger.info(f"Loading traces from {fp_cache}")
            return pd.read_pickle(fp_cache, compression="zstd")

    if len(df) == 0:
        return pd.DataFrame(columns=LOG_COLUMNS + ["n_parser_invoc"], index=df.index)

    # Start the server before creating workers.
    ensure_server(config.trace_type)

    # Avoid idle workers.
    n_workers = max(1, min(n_workers, len(df)))

    probe = SQLConnector(
        user=config.mysql_info.user,
        pwd=config.mysql_info.password,
        socket_path=config.mysql_info.socket_path,
        database=config.mysql_info.database,
    )
    try:
        log_mode = detect_log_mode(config.mysql_info.datadir_path, probe)
    finally:
        if probe.cnx is not None and probe.cnx.is_connected():
            probe.cnx.close()

    if log_mode == MODE_LEGACY and n_workers > 1:
        raise GaurServerError(
            "This instrumented server writes a single shared gaur.log, so "
            "concurrent clients would overwrite each other's traces. Rebuild "
            "the server from gaur-instrumented-apps to collect in parallel, or "
            "call get_traces_from_df with n_workers=1."
        )

    if n_workers == 1:
        # Keep the single-worker path in this process.
        chunk_positions = [np.arange(len(df))]
    else:
        # Strided chunks keep similar workloads distributed across workers.
        n_chunks = min(len(df), n_workers * CHUNKS_PER_WORKER)
        chunk_positions = [np.arange(k, len(df), n_chunks) for k in range(n_chunks)]

    n_chunks = len(chunk_positions)
    progress = tqdm(total=len(df), disable=disable_tqdm)
    results: dict[int, pd.DataFrame] = {}

    # Do not let workers inherit a parent connection.
    _close_worker_conn()

    try:
        if n_chunks == 1:
            results[0] = _collect_chunk(
                df,
                _partial_path(cache_dir, str_hash_df, 0, 1),
                use_cache,
                log_mode,
                lambda: progress.update(1),
            )
        else:
            logger.info(
                f"Collecting with {n_workers} processes over {n_chunks} chunks"
            )
            tasks = [
                (
                    k,
                    df.iloc[chunk_positions[k]],
                    _partial_path(cache_dir, str_hash_df, k, n_chunks),
                    use_cache,
                    log_mode,
                )
                for k in range(n_chunks)
            ]
            # Workers must inherit the configured process state.
            ctx = multiprocessing.get_context("fork")
            pool = ctx.Pool(processes=n_workers, initializer=_worker_init)
            try:
                for chunk_id, df_chunk in pool.imap_unordered(
                    _collect_chunk_task, tasks
                ):
                    results[chunk_id] = df_chunk
                    # Progress is reported per completed chunk.
                    progress.update(len(chunk_positions[chunk_id]))
                pool.close()
            except BaseException:
                # Stop siblings before propagating the error.
                pool.terminate()
                raise
            finally:
                pool.join()
    finally:
        progress.close()
        # Close the in-process connection, if any.
        _close_worker_conn()

    # Restore input order and index after chunks complete out of order.
    frames = []
    positions = []
    for chunk_id in sorted(results):
        df_chunk = results[chunk_id]
        if df_chunk.empty:
            continue
        frames.append(df_chunk)
        positions.append(chunk_positions[chunk_id][: len(df_chunk)])

    df_traces = pd.concat(frames, ignore_index=True)
    pos = np.concatenate(positions)
    order = np.argsort(pos, kind="stable")
    df_traces = df_traces.iloc[order]
    df_traces.index = df.index[pos[order]]

    # Remove queries for which no semantic tree was collected.

    missing_mask = df_traces["semantic_tree"].isnull()
    num_missing = missing_mask.sum()

    if num_missing > 0:
        logger.critical(f"Removed {num_missing} queries with missing semantic_tree.")
        df_traces = df_traces[~missing_mask]
    else:
        logger.info("Successfully collected a semantic_tree for each query.")

    if use_cache:
        df_traces.to_pickle(fp_cache, compression="zstd")
        for k in range(n_chunks):
            fp_partial = _partial_path(cache_dir, str_hash_df, k, n_chunks)
            if os.path.isfile(fp_partial):
                os.remove(fp_partial)
                logger.info(f"Removed partial cache file {fp_partial}")

    return df_traces

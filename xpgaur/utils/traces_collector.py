import hashlib
import os

import mysql
import xpgaur.config as config
import logging
import pandas as pd
import mysql.connector
from tqdm import tqdm

from .mysql_wrapper import SQLConnector

logger = logging.getLogger(__name__)


class GaurTraceCollector:
    def __init__(self, fp_datadir: str):
        self.fp_log = fp_datadir + "gaur.log"

    def reset_logfile(self) -> None:
        """Empty the Gaur log file."""
        with open(self.fp_log, "w") as file:
            file.write(
                "query_id,n_terminal,n_nonterminal,is_syntax_error,semantic_tree,depth\n"
            )

    def collect_logfile(self) -> pd.DataFrame:
        """Retrieve traces in Gaur log file.

        Returns:
            pd.DataFrame: Traces
        """
        df = pd.read_csv(self.fp_log)
        self.reset_logfile()
        return df


def merge_traces(df_traces: pd.DataFrame, query: str) -> pd.DataFrame:
    """Function merging multiple GAUR traces into

    Args:
        df_traces (pd.DataFrame): _description_
        query (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Columns to handle:
    # "query_id": Array
    # "n_terminal": Sum
    # "n_nonterminal": Sum
    # "is_syntax_error": OR
    # "semantic_tree": Merge (split on "||-||"), add node information left, and add
    #  right the edges.
    # "depth": Sum:
    # "n_parser_invoc": Count.

    # Stacked queries lead to multiple query_id, but this field is more for tracking
    # traces than prediction. We simply take the value of first DataFrame entry.
    qid = df_traces.iloc[0]["query_id"]
    n_terminal = df_traces["n_terminal"].sum()
    n_nonterminal = df_traces["n_nonterminal"].sum()
    depth = df_traces["depth"].sum()
    n_parser_invoc = df_traces.shape[0]
    is_syntax_error = (df_traces["is_syntax_error"] == 1).any()

    nodes = []
    edges = []

    for tree in df_traces["semantic_tree"]:
        # Trace is split between a nodes part and an edges part.
        if isinstance(tree, str):
            parts = tree.split("||-||")
            if len(parts) == 2:
                n, e = parts
                nodes.append(n)
                edges.append(e)
            else:
                logger.warning(f"Unexpected split output for query {query}: {parts}")
        else:
            # With some queries, a trace is empty. Maybe a bug, for now just ignore them
            logger.warning(f"A trace is invalid for query: {query}")

    # Now merge all
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
    """Executes a SQL query and returns associated GAUR trace as a DataFrame.

    Initializes the SQL connection and resets logs if needed. On failure,
    returns a DataFrame with missing values. Merges traces if multiple
    parser invocations are detected.

    Args:
        query (str): SQL query to execute.
        sqlc (SQLConnector): Connector for running the query.
        gtc (GaurTraceCollector): Collector for trace logs.

    Returns:
        pd.DataFrame: Trace log data or NA-filled DataFrame on failure.
    """

    if sqlc.cnx is None or not sqlc.cnx.is_connected():
        # Initializing a new connection lead to parser invocations: we purge the
        # logfile afterwards.
        sqlc.init_new_cnx()
        gtc.reset_logfile()

    # TODO: Some queries takes an infinite time to execute. Maybe running pt-kill in
    # the back can resolve that, but we observe cases where the server crashes.
    # To prevent that, I could probably implement a timer mechanism:
    # - After 1min without return, tries to collect trace (it was probably produced,
    # it's only the execution that takes forever).
    # - Create a root connection and restart the server using the RESTART statement.
    # - Wait a bit for the restart and go to the next query.
    retcode = sqlc.execute_query(query)

    if retcode == 1:
        gtc.reset_logfile()

        # TODO: How to better handle this ? For now we return dataframes with NA values
        # so that the according row can easily be removed.
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

    # If multiple parser invocation: merge them
    if df_traces.shape[0] > 1:
        return merge_traces(df_traces=df_traces, query=query)

    df_traces["n_parser_invoc"] = 1
    return df_traces


def get_traces_from_df(
    df: pd.DataFrame,
    use_cache: bool = True,
    use_datadir: bool = True,
    disable_tqdm : bool = False,
) -> pd.DataFrame:
    """Return the GAUR traces associated to the passed DataFrame.


    We need to collect traces for all samples. This is a very long process.
    Hence, we compute a unique identifier, for  the resulting DataFrame  and collect
    or load traces if already cached given the identifier. get_traces_from_df takes
    care of that.

    Args:f
        df (pd.DataFrame): DataFrame with SQL queries to get traces from.
        use_cache (bool):
    Returns:
        pd.DataFrame: Traces
    """

    # Uses a cache mechanism by computing the DataFrame hash. Try to locate pickle.
    # If not present, compute traces.
    str_hash_df = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

    ltraces = []
    start_idx = 0

    # We save the partial each 10% of traces collected.
    save_interval = max(1, len(df) // 10)  # every 10%
    
    if use_datadir:
        fp_cache_partial = "".join(
            [config.ppths.cache_datadir_path, str_hash_df, "-partial.pkl"]
        )
    else:
        fp_cache_partial = "".join(
            [config.ppths.cache_path, str_hash_df, "-partial.pkl"]
        )

    if use_cache:
        # Attempt to load full file if it exists.
        if use_datadir:
            fp_cache = "".join(
                [
                    config.ppths.cache_datadir_path,
                    str_hash_df,
                    ".pkl",
                ]
            )
        else:
            fp_cache = "".join([config.ppths.cache_path, str_hash_df, ".pkl"])

        if os.path.isfile(fp_cache):
            logger.info(f"Loading traces from {fp_cache}")
            return pd.read_pickle(fp_cache)

        # Or attempt to load partial file is it exists.


        if os.path.isfile(fp_cache_partial):
            logger.info(f"Loading partial results from {fp_cache_partial}")
            partial_df = pd.read_pickle(fp_cache_partial)
            ltraces.append(partial_df)
            start_idx = len(partial_df)
            logger.info(f"Resuming from row {start_idx}/{len(df)}")

    sqlc = SQLConnector(
        user=config.mysql_info.user,
        pwd=config.mysql_info.password,
        socket_path=config.mysql_info.socket_path,
        database=config.mysql_info.database,
    )
    gtc = GaurTraceCollector(fp_datadir=config.mysql_info.datadir_path)

    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df),disable=disable_tqdm)):
        # We want to catch mysql.connector.erors.InterfaceError here.
        # When that happens, save all collected trace so far, so we don't have
        # to restart from scratch.
        if i < start_idx:
            continue
        try:
            ltraces.append(get_traces_from_query(row.full_query, sqlc, gtc))

        except mysql.connector.errors.InterfaceError as e:
            logger.error(f"MySQL interface error on row {i}: {e}")
            if use_cache and ltraces:
                partial_df = pd.concat(ltraces)
                partial_df.index = df.index[: len(partial_df)]
                pd.to_pickle(partial_df, fp_cache_partial)
                logger.info(f"Partial results saved to {fp_cache_partial}")
            raise  # Re-raise so caller knows something went wrong

        # Checkpoint every 10% of data processed
        if use_cache and i % save_interval == 0:
            partial_df = pd.concat(ltraces)
            partial_df.index = df.index[: len(partial_df)]
            pd.to_pickle(partial_df, fp_cache_partial)
            logger.info(f"Progress checkpoint saved at row {i+1}/{len(df)}")

    df_traces = pd.concat(ltraces)

    # Preserve index
    df_traces.index = df.index

    # Queries for which we didn't manage to collect traces are removed. Most often,
    # this is caused because an error is thrown in mysql_wrapper that is not caught.
    # Ultimately, we should be able to catch all errors caused by the queries in the
    # dataset, and hence, not removing any entry.

    missing_mask = df_traces["semantic_tree"].isnull()
    num_missing = missing_mask.sum()

    if num_missing > 0:
        logger.critical(f"Removed {num_missing} queries with missing semantic_tree.")
        df_traces = df_traces[~missing_mask]
    else:
        logger.info("Successfully collected a semantic_tree for each query.")

    if use_cache:
        df_traces.to_pickle(fp_cache)

    return df_traces

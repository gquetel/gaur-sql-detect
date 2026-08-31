"""Check that parallel collection keeps every trace next to its query.

The collector may process these rows in separate workers:

    input:  [query A, query B]
    result: [trace for A, trace for B]

Even if query B finishes first, its trace must not become the first result. This set
of tests ensure this is the case without using actual Gaur connectors.
"""

import multiprocessing
import os
import time

import pandas as pd
import pytest

import gaur_sqld.utils.traces_collector as tc


def _make_trace_for_query(query: str) -> pd.DataFrame:
    """Return one trace whose semantic tree identifies its source query."""
    return pd.DataFrame(
        {
            "query_id": [sum(map(ord, query)) % 100000],
            "n_terminal": [len(query)],
            "n_nonterminal": [query.count(" ")],
            "is_syntax_error": [False],
            "semantic_tree": [f"tree({query})"],
            "depth": [len(query) % 7],
            "n_parser_invoc": [1],
        },
        index=[0],
    )


class FakeConnector:
    """Small connector substitute that never talks to MySQL."""

    def __init__(self, **kwargs):
        self.cnx = None

    def init_new_cnx(self):
        self.cnx = self

    def is_connected(self):
        return True

    def close(self):
        self.cnx = None

    @property
    def connection_id(self):
        # Give each worker a distinct log filename.
        return os.getpid()

    def execute_query(self, _query):
        return 0


class FakeTraceCollector:
    """Trace collector substitute that records worker log-file cleanup."""

    def __init__(self, fp_datadir, connection_id=None):
        self.fp_datadir = fp_datadir
        self.connection_id = None
        self.fp_log = os.devnull
        self.bind(connection_id)

    def bind(self, connection_id):
        self.connection_id = connection_id
        if connection_id is not None:
            self.fp_log = os.path.join(
                self.fp_datadir, f"gaur.{connection_id}.log"
            )
            with open(self.fp_log, "w"):
                pass

    def reset_logfile(self):
        pass

    def cleanup(self):
        if self.connection_id is not None:
            try:
                os.remove(self.fp_log)
            except FileNotFoundError:
                pass


def _patch_collector_with_test_doubles(
    monkeypatch, *, log_dir=None, make_chunks_finish_out_of_order=False
):
    """Patch the collector so tests exercise scheduling, not MySQL."""
    if log_dir is not None:
        monkeypatch.setattr(tc.config.mysql_info, "datadir_path", str(log_dir))

    monkeypatch.setattr(tc, "ensure_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tc, "detect_log_mode", lambda *args, **kwargs: tc.MODE_PER_CONNECTION
    )
    monkeypatch.setattr(tc, "SQLConnector", FakeConnector)
    monkeypatch.setattr(tc, "GaurTraceCollector", FakeTraceCollector)

    def trace_for_query(query, _sql_connector, _trace_collector):
        if make_chunks_finish_out_of_order:
            query_number = int(query.split()[1])
            # Lower-numbered queries take longer, so chunks finish out of order.
            time.sleep(max(0, 16 - query_number) * 0.002)
        return _make_trace_for_query(query)

    monkeypatch.setattr(tc, "get_traces_from_query", trace_for_query)


def _make_queries(count):
    return [f"SELECT {i} FROM t WHERE x = {i * 3}" for i in range(count)]


def _collect_without_cache(df, n_workers):
    return tc.get_traces_from_df(
        df, use_cache=False, disable_tqdm=True, n_workers=n_workers
    )


@pytest.mark.parametrize("n_workers", [2, 4])
def test_parallel_collection_keeps_each_trace_with_its_query(
    monkeypatch, n_workers, tmp_path
):
    """Parallel results stay aligned when chunks finish out of order."""
    _patch_collector_with_test_doubles(
        monkeypatch,
        log_dir=tmp_path,
        make_chunks_finish_out_of_order=True,
    )
    queries = _make_queries(16)
    index = pd.Index([(i * 7919) % 10007 for i in range(16)], name="input_id")
    df = pd.DataFrame({"full_query": queries}, index=index)

    traces = _collect_without_cache(df, n_workers)

    assert traces.index.equals(df.index)
    assert traces["semantic_tree"].tolist() == [f"tree({q})" for q in queries]


def test_parallel_collection_handles_duplicate_input_indexes(monkeypatch, tmp_path):
    """Duplicate input indexes do not make result alignment ambiguous."""
    _patch_collector_with_test_doubles(
        monkeypatch,
        log_dir=tmp_path,
        make_chunks_finish_out_of_order=True,
    )
    queries = [f"SELECT {i}" for i in range(20)]
    df = pd.DataFrame({"full_query": queries}, index=[0, 1] * 10)

    traces = _collect_without_cache(df, n_workers=4)

    assert traces.index.equals(df.index)
    assert traces["semantic_tree"].tolist() == [f"tree({q})" for q in queries]


def test_worker_count_validation_and_auto_selection():
    """The automatic setting stays serial for short collections."""
    assert tc._resolve_n_workers(-1, tc.DEFAULT_PARALLEL_MIN_ROWS) == 1
    assert tc._resolve_n_workers(-1, tc.DEFAULT_PARALLEL_MIN_ROWS + 1) == (
        os.cpu_count() or 1
    )

    for invalid_count in (0, -2):
        with pytest.raises(ValueError):
            tc._resolve_n_workers(invalid_count, 10)


def test_workers_remove_log_files_after_success_and_failure(monkeypatch, tmp_path):
    """Worker log files are removed after normal and failed pool shutdown."""
    _patch_collector_with_test_doubles(monkeypatch, log_dir=tmp_path)
    df = pd.DataFrame({"full_query": _make_queries(64)})

    # Normal pool shutdown and forced termination use different cleanup paths.
    _collect_without_cache(df, n_workers=4)
    assert list(tmp_path.iterdir()) == []

    all_workers_started = multiprocessing.Barrier(4)

    def trace_for_query_or_fail(query, _sql_connector, _trace_collector):
        query_number = int(query.split()[1])
        if query_number < 4:
            # Let every worker create its log before the failing task starts.
            all_workers_started.wait(timeout=10)
        if query_number == 4:
            raise RuntimeError("simulated collection failure")
        return _make_trace_for_query(query)

    monkeypatch.setattr(tc, "get_traces_from_query", trace_for_query_or_fail)
    with pytest.raises(RuntimeError, match="simulated collection failure"):
        _collect_without_cache(df, n_workers=4)

    assert list(tmp_path.iterdir()) == []

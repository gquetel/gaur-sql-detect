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


class _StubConnector:
    """Minimal already-connected SQLConnector stand-in."""

    def __init__(self):
        self.cnx = self

    def is_connected(self):
        return True

    def execute_query(self, _query):
        return 0


class _StubCollectorEmptyLog:
    """GaurTraceCollector stand-in for a query that logs no parser invocation."""

    connection_id = None

    def collect_logfile(self):
        return pd.DataFrame(columns=tc.LOG_COLUMNS)


def test_query_with_empty_trace_frame_becomes_failed_row():
    """A query that runs but logs no parser invocation still yields one row."""
    trace = tc.get_traces_from_query(
        "SELECT 1", _StubConnector(), _StubCollectorEmptyLog()
    )

    assert len(trace) == 1
    assert trace["n_parser_invoc"].iloc[0] == 0
    assert trace["semantic_tree"].iloc[0] == "||-||"


def test_parallel_collection_keeps_alignment_when_some_queries_fail(
    monkeypatch, tmp_path
):
    """A query yielding an empty trace frame does not shift later rows out of place."""
    _patch_collector_with_test_doubles(monkeypatch, log_dir=tmp_path)

    queries = _make_queries(16)
    # Every third query ran but logged no parser invocation.
    failing = {q for i, q in enumerate(queries) if i % 3 == 0}

    def trace_for_query(query, _sql_connector, _trace_collector):
        if query in failing:
            return tc._failed_trace_frame()
        return _make_trace_for_query(query)

    monkeypatch.setattr(tc, "get_traces_from_query", trace_for_query)

    df = pd.DataFrame({"full_query": queries})
    traces = _collect_without_cache(df, n_workers=4)

    assert traces.index.equals(df.index)
    expected = ["||-||" if q in failing else f"tree({q})" for q in queries]
    assert traces["semantic_tree"].tolist() == expected


def test_merge_traces_combines_rows_and_avoids_spurious_warning(caplog):
    """Multiple parser invocations merge into one row, and the empty-tree
    marker ("||-||") is not mistaken for an invalid split."""
    df_traces = pd.DataFrame(
        {
            "query_id": [1, 1],
            "n_terminal": [2, 1],
            "n_nonterminal": [1, 0],
            "is_syntax_error": [0, 1],
            "semantic_tree": ["NODEA||-||EDGEA", "||-||"],
            "depth": [3, 0],
        }
    )

    with caplog.at_level("WARNING", logger=tc.logger.name):
        merged = tc.merge_traces(df_traces=df_traces, query="SELECT 1; SELECT 2")

    assert merged["n_terminal"].iloc[0] == 3
    assert merged["n_nonterminal"].iloc[0] == 1
    assert merged["depth"].iloc[0] == 3
    assert merged["n_parser_invoc"].iloc[0] == 2
    assert merged["semantic_tree"].iloc[0] == "NODEA||-||EDGEA"
    assert merged["is_syntax_error"].iloc[0] == 1
    assert pd.api.types.is_integer_dtype(merged["is_syntax_error"])
    assert not pd.api.types.is_bool_dtype(merged["is_syntax_error"])
    assert "Unexpected split output" not in caplog.text


def test_chunk_length_mismatch_raises(monkeypatch, tmp_path):
    """A chunk returning the wrong row count is a hard error, not absorbed."""
    _patch_collector_with_test_doubles(monkeypatch, log_dir=tmp_path)
    monkeypatch.setattr(tc, "_collect_chunk", lambda *a, **k: pd.DataFrame())

    df = pd.DataFrame({"full_query": _make_queries(4)})

    with pytest.raises(RuntimeError, match="trace rows"):
        _collect_without_cache(df, n_workers=1)


def test_collect_chunk_checkpoint_count_mismatch_raises(monkeypatch, tmp_path):
    """A misbehaving get_traces_from_query is caught before it corrupts the
    partial-results cache, instead of silently mis-sizing the checkpoint."""
    _patch_collector_with_test_doubles(monkeypatch, log_dir=tmp_path)

    def bad_trace_for_query(query, _sql_connector, _trace_collector):
        # The exact bug this fix closes: not exactly one row per query.
        return pd.concat(
            [_make_trace_for_query(query), _make_trace_for_query(query)],
            ignore_index=True,
        )

    monkeypatch.setattr(tc, "get_traces_from_query", bad_trace_for_query)

    chunk = pd.DataFrame({"full_query": _make_queries(3)})
    fp_partial = str(tmp_path / "partial.pkl")

    with pytest.raises(RuntimeError, match="trace rows"):
        tc._collect_chunk(
            chunk, fp_partial, use_cache=True, log_mode=tc.MODE_PER_CONNECTION
        )


def test_collect_logfile_reads_real_file(tmp_path):
    """collect_logfile parses real CSV content, including the empty-tree
    case and a header-only file with no logged parser invocation."""
    gtc = tc.GaurTraceCollector(fp_datadir=str(tmp_path))
    with open(gtc.fp_log, "w", newline="") as f:
        f.write(tc.LOG_HEADER)
        f.write("42,3,1,0,NODE||-||EDGE,2\n")
        f.write("43,0,0,1,,0\n")

    df = gtc.collect_logfile()

    assert df["query_id"].tolist() == [42, 43]
    assert df["semantic_tree"].tolist() == ["NODE||-||EDGE", "||-||"]
    assert df["is_syntax_error"].tolist() == [0, 1]
    assert pd.api.types.is_integer_dtype(df["is_syntax_error"])

    # The log file is reset (header-only) after reading.
    with open(gtc.fp_log) as f:
        assert f.read() == tc.LOG_HEADER

    # A header-only file (no parser invocation logged) round-trips to 0 rows.
    assert len(gtc.collect_logfile()) == 0


def test_cache_round_trip_preserves_length(monkeypatch, tmp_path):
    """Traces reloaded from cache have the same length as a fresh collection."""
    _patch_collector_with_test_doubles(monkeypatch, log_dir=tmp_path)
    monkeypatch.setattr(tc.config.ppths, "base_path", str(tmp_path / "project"))

    df = pd.DataFrame({"full_query": _make_queries(8)})

    first = tc.get_traces_from_df(df, use_cache=True, disable_tqdm=True, n_workers=1)
    second = tc.get_traces_from_df(df, use_cache=True, disable_tqdm=True, n_workers=1)

    assert len(first) == len(df)
    assert len(second) == len(first)
    assert second["semantic_tree"].tolist() == first["semantic_tree"].tolist()


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

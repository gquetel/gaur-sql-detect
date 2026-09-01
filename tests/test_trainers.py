"""Tests for the pure/near-pure logic in trainers.py.

Most of trainers.py is heavy ML-training orchestration (real torch/sklearn
models, file I/O, plotting) that is integration-level and out of scope here.
This file covers the isolated pieces whose contracts a subtle bug could break
silently: split routing, threshold computation, model/trace-type selection,
the anomaly-score sign convention, and the scoring batch loop.
"""

import argparse

import numpy as np
import pandas as pd
import pytest

import gaur_sqld.utils.trainers as trainers


def test_get_traces_per_split_raises_on_length_mismatch(monkeypatch):
    """A trace collector returning the wrong row count is a hard error."""
    monkeypatch.setattr(
        trainers, "get_traces_from_df", lambda df, use_cache: df.iloc[:0]
    )

    df = pd.DataFrame(
        {"full_query": ["SELECT 1", "SELECT 2"], "split": ["train", "train"]}
    )

    with pytest.raises(RuntimeError, match="Collected"):
        trainers.get_traces_per_split(df, use_cache=False)


def test_get_traces_per_split_routes_and_merges_rows(monkeypatch):
    """Each row lands in the output split matching its 'split' label, merged
    with its own trace columns (not a neighbor's), and use_cache is forwarded.

    A wrong filter, a shuffled concat, or a dropped use_cache argument would
    all pass the length-mismatch guard above while silently corrupting the
    train/test/val split that every downstream model gets trained/scored on.
    """
    seen_use_cache = []

    def fake_get_traces(df, use_cache):
        seen_use_cache.append(use_cache)
        return pd.DataFrame(
            {"trace_col": [f"trace-{i}" for i in df.index]}, index=df.index
        )

    monkeypatch.setattr(trainers, "get_traces_from_df", fake_get_traces)

    df = pd.DataFrame(
        {
            "full_query": ["Q0", "Q1", "Q2", "Q3"],
            "split": ["train", "test", "val", "train"],
        }
    )

    df_train, df_test, df_val = trainers.get_traces_per_split(df, use_cache=True)

    assert df_train["full_query"].tolist() == ["Q0", "Q3"]
    assert df_train["trace_col"].tolist() == ["trace-0", "trace-3"]
    assert df_test["full_query"].tolist() == ["Q1"]
    assert df_test["trace_col"].tolist() == ["trace-1"]
    assert df_val["full_query"].tolist() == ["Q2"]
    assert df_val["trace_col"].tolist() == ["trace-2"]
    assert seen_use_cache == [True, True, True]


def test_get_threshold_for_max_rate_bounds_the_exceedance_rate():
    """The threshold must admit roughly `max_rate` of scores above it, and a
    looser rate must lower (not raise) the threshold.

    Getting the percentile direction backwards (e.g. `max_rate * 100` instead
    of `(1 - max_rate) * 100`) would silently invert the anomaly-detection
    decision boundary used for every downstream FPR/precision number.
    """
    s_val = np.arange(0, 101)  # 101 unique values, 0..100 -> no interpolation.

    threshold = trainers.get_threshold_for_max_rate(s_val, max_rate=0.01)
    assert threshold == 99
    assert np.sum(s_val > threshold) == 1  # only the top score exceeds it.

    looser_threshold = trainers.get_threshold_for_max_rate(s_val, max_rate=0.10)
    assert looser_threshold < threshold


def test_select_models_expands_named_group():
    """A group name expands to its member models, and nothing else."""
    args = argparse.Namespace(models=["gaur"])
    selected = trainers.select_models(args)
    assert set(selected) == {"ocsvm", "ae"}
    assert all(callable(fn) for fn in selected.values())


def test_select_models_all_returns_full_registry():
    """'all' selects every registered model, not just the default group."""
    args = argparse.Namespace(models=["all"])
    selected = trainers.select_models(args)
    assert set(selected) == {"ocsvm", "ae", "ocsvm_ruleid", "ae_ruleid"}


def test_select_models_skips_unknown_and_logs_warning(caplog):
    """An unrecognized model name is dropped, not silently ignored or crashed
    on, and a warning is logged so a typo'd CLI arg is noticed."""
    args = argparse.Namespace(models=["ocsvm", "not_a_model"])
    with caplog.at_level("WARNING", logger=trainers.logger.name):
        selected = trainers.select_models(args)
    assert set(selected) == {"ocsvm"}
    assert "not_a_model" in caplog.text


def test_select_trace_types_all_returns_full_list():
    """'all' must stay in sync with the module's hardcoded valid-traces list.

    The two lists are maintained separately in the source; a future edit to
    one without the other would only show up here.
    """
    args = argparse.Namespace(trace_type=["all"])
    assert trainers.select_trace_types(args) == [
        "expert",
        "claude",
        "chatgpt",
        "llama",
        "mistral",
        "gpt-oss",
    ]


def test_select_trace_types_removes_all_invalid_entries():
    """Two consecutive unrecognized trace types should both be dropped."""
    args = argparse.Namespace(trace_type=["expert", "bogus1", "bogus2", "claude"])
    result = trainers.select_trace_types(args)
    assert result == ["expert", "claude"]


class _FakeClf:
    """Stand-in for `model.clf` exposing only `decision_function`."""

    def __init__(self, dists):
        self._dists = np.asarray(dists)
        self.received_kwargs = None

    def decision_function(self, X, **kwargs):
        self.received_kwargs = kwargs
        return self._dists


class _FakeModel:
    def __init__(self, dists):
        self.clf = _FakeClf(dists)


def test_decision_score_generic_negates_distance():
    """A positive clf distance (inlier) must become a negative anomaly score
    and vice versa; getting this sign backwards silently inverts every
    precision/recall metric computed from it."""
    model = _FakeModel([1.0, -2.0, 3.0])
    scores = trainers.decision_score_generic(model, X=np.zeros((3, 1)))
    assert list(scores) == [-1.0, 2.0, -3.0]


def test_decision_score_ae_negates_distance_and_requests_tensor_scoring():
    """The AE variant applies the same sign flip and must call
    decision_function with is_tensor=True (its documented calling
    convention), not the plain-array path used by the other models."""
    model = _FakeModel([0.5, -0.5])
    scores = trainers.decision_score_ae(model, X="tensor-placeholder")
    assert list(scores) == [-0.5, 0.5]
    assert model.clf.received_kwargs == {"is_tensor": True}


def test_get_scores_generic_batched_matches_unbatched():
    """Batched and non-batched scoring must return identical labels/scores,
    in the original row order.

    Uses a batch size that does not evenly divide the row count, so an
    off-by-one in the last partial batch or a wrong concatenation order would
    only show up in this comparison.
    """
    df = pd.DataFrame({"value": np.arange(10), "label": [i % 2 for i in range(10)]})

    def preprocess(model, batch_df, use_scaler=False):
        return batch_df["value"].to_numpy(), batch_df["label"].tolist()

    def score(model, X):
        return X * 10

    labels_full, scores_full = trainers.get_scores_generic(
        df=df, model=None, preprocess_fn=preprocess, score_fn=score, batch_size=None
    )
    labels_batched, scores_batched = trainers.get_scores_generic(
        df=df, model=None, preprocess_fn=preprocess, score_fn=score, batch_size=3
    )

    expected_labels = df["label"].tolist()
    expected_scores = (df["value"] * 10).tolist()
    assert labels_full.tolist() == expected_labels
    assert labels_batched.tolist() == expected_labels
    assert scores_full.tolist() == expected_scores
    assert scores_batched.tolist() == expected_scores

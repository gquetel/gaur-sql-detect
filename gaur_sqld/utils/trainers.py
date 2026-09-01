import argparse
from datetime import datetime
import os
from typing import Any, Callable
import numpy as np
import random
import pandas as pd
import logging
import torch
from tqdm import tqdm

import gaur_sqld
import gaur_sqld.config as config
from gaur_sqld.models.Gaur import AutoEncoder_Gaur, LOF_Gaur, OCSVM_Gaur

from .traces_collector import get_traces_from_df
from gaur_sqld.server import stop_server
from .explain import (
    get_balanced_accuracy_per_attack,
    get_recall_per_attack,
    get_recall_per_statement_type,
    plot_pr_curves_plt_from_scores,
    plot_roc_curves_plt_from_scores,
    get_metrics_treshold,
)


logger = logging.getLogger(__name__)


def get_traces_per_split(
    df: pd.DataFrame,
    use_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collect traces per split to enable cross-run cache reuse.
    
    We need to collect traces for all samples. This is a very long process. Hence, we
    compute a unique identifier, for  the resulting DataFrame  and collect or load 
    traces if already cached given the identifier. Because in some cases we train on 
    different dataset, but test on the same, we split the loading / saving of the
    splits. This prevent collecting all traces when only the train set changes.
    """
    l_traces = []
    for split_name in ["train", "test", "val"]:
        df_split = df[df["split"] == split_name]
        if len(df_split) == 0:
            continue
        traces = get_traces_from_df(df=df_split, use_cache=use_cache)
        if len(traces) != len(df_split):
            raise RuntimeError(
                f"Collected {len(traces)} traces for {len(df_split)} rows in "
                f"the '{split_name}' split; every input row must yield "
                "exactly one trace row."
            )
        merged = pd.concat([traces, df_split], axis=1)
        l_traces.append(merged)

    df_merged = pd.concat(l_traces)
    return (
        df_merged[df_merged["split"] == "train"],
        df_merged[df_merged["split"] == "test"],
        df_merged[df_merged["split"] == "val"],
    )


max_num_threads = 64
n_jobs = min(max_num_threads, int(os.cpu_count() * 0.9))

training_results = []
inference_times = []


def init_device() -> torch.device:
    """Initialize the device to use for experiments

    Returns:
        torch.device: device to use
    """
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    if USE_CUDA:
        logger.info(
            "Using device: %s for experiments.", torch.cuda.get_device_name()
        )
        torch.cuda.set_per_process_memory_fraction(0.99, 0)
    else:
        logger.info("Using CPU for experiments.")
    return device


def get_threshold_for_max_rate(s_val, max_rate=0.001):
    """Compute threshold given a max allowed FPR.

    Args:
        s_val (_type_): _description_
        max_rate (float, optional): _description_. Defaults to 0.00001.

    Returns:
        _type_: _description_
    """
    s_val = np.array(s_val)
    percentile = (1 - max_rate) * 100
    return np.percentile(s_val, percentile)


def init_seeds():
    seed = gaur_sqld.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_scores_generic(
    df: pd.DataFrame,
    model,
    preprocess_fn: Callable[[Any, pd.DataFrame], tuple[np.ndarray, np.ndarray]],
    score_fn: Callable[[Any, np.ndarray], np.ndarray],
    batch_size: int | None = None,
    use_scaler: bool = False,
    measure_time: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic scoring loop."""
    all_labels, all_scores = [], []
    a = datetime.now()

    if batch_size:
        for start_idx in tqdm(range(0, len(df), batch_size), desc="Scoring batches"):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            X, labels = preprocess_fn(model, batch_df, use_scaler=use_scaler)
            scores = score_fn(model, X)

            all_labels.extend(labels)
            all_scores.extend(scores)

    else:
        X, all_labels = preprocess_fn(model, df, use_scaler=use_scaler)
        all_scores = score_fn(model, X)

    b = datetime.now()
    inference_time = b - a

    if measure_time:
        # The function is called for validation and test.
        # We choose to only study inference time of test.
        logger.info(f"Inference took {inference_time} seconds.")
        inference_times.append(
            {
                "model": model.model_name,
                "inference_time": inference_time,
                "n_samples": len(all_labels),
            }
        )

    return np.array(all_labels), np.array(all_scores)


def decision_score_generic(model: OCSVM_Gaur | LOF_Gaur, X: np.ndarray):
    # dists are a distance to the separating hyperplane.
    # Negative distance is an outlier (attack)
    # Positive distance is an inlier (normal)
    dists = model.clf.decision_function(X)

    # Process dists so that positive class is > 0 as asked by
    # average_precision_score & roc_auc_score
    return -dists


def decision_score_ae(model: AutoEncoder_Gaur, X: np.ndarray):
    # dists are a distance to the separating hyperplane.
    # Negative distance is an outlier (attack)
    # Positive distance is an inlier (normal)
    dists = model.clf.decision_function(X, is_tensor=True)

    # Process dists so that positive class is > 0 as asked by
    # average_precision_score & roc_auc_score
    return -dists


def preprocess_gaur(
    model: OCSVM_Gaur | LOF_Gaur, df: pd.DataFrame, use_scaler: bool = False
):
    """Transform dataframe into features ready to be scored and their labels.

    Args:
        model (OCSVM_Gaur | LOF_Gaur): Model object to use to preprocess.
        df (pd.DataFrame): Input dataframe
        use_scaler (bool, optional): Whether to use scaler or not. Defaults to False.

    Returns:
        _type_: _description_
    """
    df_pped, labels = model.preprocess_for_preds(df=df, drop_og_columns=False)

    # 1 => Probas (ppeds only)

    # For GAUR, some of the  original columns are features (n_terminal
    # for instance) so we set this manual list of features to drop.

    X_clean = df_pped.drop(
        [
            "label",
            "full_query",
            "statement_type",
            "query_template_id",
            "user_inputs",
            "attack_id",
            "attack_technique",
            "split",
            "attack_status",
            "attack_stage",
            "tamper_method",
        ],
        axis=1,
    ).to_numpy()

    labels = df["label"].tolist()

    if use_scaler:
        X_scaled = model._scaler.transform(X_clean)
        return X_scaled, labels

    return X_clean, labels


def preprocess_gaur_ae(model: AutoEncoder_Gaur, df: pd.DataFrame, use_scaler: Any):
    """
    Preprocessing function for GAUR-based pipelines.
    Returns:
        X (np.ndarray): Features (as tensors) ready for scoring.
        labels (list[int]): Ground truth labels.
        use_scaler: Unused here, but required to fit to compute_metric_generic
    """
    df_pped, labels = model.preprocess_for_preds(df=df, drop_og_columns=False)

    # 1 => Probas (ppeds only)

    # For GAUR, some of the  original columns are features (n_terminal
    # for instance) so we set this manual list of features to drop.

    X_clean = df_pped.drop(
        [
            "label",
            "full_query",
            "statement_type",
            "query_template_id",
            "user_inputs",
            "attack_id",
            "attack_technique",
            "split",
            "attack_status",
            "attack_stage",
            "tamper_method",
        ],
        axis=1,
    )

    labels = df["label"].tolist()
    tensors = model._dataframe_to_tensor_batched(X_clean, batch_size=4096)
    tensors = tensors.to(model.device)

    return tensors, labels


def compute_metrics_generic(
    model,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    model_name: str,
    preprocess_fn: Callable,
    get_decision_scores_fn: Callable,
    use_scaler: bool,
    insider_as_fn: bool = False,
    use_batches: bool = False,
    fixed_fpr: float = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generic function to evaluate an anomaly detection model by computing prediction scores,
    selecting a decision threshold, and logging evaluation metrics.

    Args:
        model: Trained anomaly detection model.
        df_test (pd.DataFrame): Test dataset containing samples and true labels.
        df_val (pd.DataFrame): Validation dataset used to compute the decision threshold.
        model_name (str): Identifier used for logging and result tracking.
        preprocess_fn (Callable): Function that takes (model, batch_df) and returns
            a tuple (X, labels), where X can be passed to the model for scoring.
        get_decision_scores_fn (Callable): Function that takes (model, X) and returns
            anomaly scores (e.g., negative distances to decision boundary).
        use_scaler (bool): Whether scaling should be used for the features.
        insider_as_fn (bool): False by default. If set to True, the queries with the
            attack_technique "insider" will be considered as False Negative: the data
            collector is not able to collect information for this attack.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Ground truth labels from the test set.
            - Computed anomaly scores for the test set.
    """
    # Get test and val scores
    batch = 4096 if use_batches else None
    l_test, s_test = get_scores_generic(
        df=df_test,
        batch_size=batch,
        model=model,
        preprocess_fn=preprocess_fn,
        score_fn=get_decision_scores_fn,
        use_scaler=use_scaler,
        measure_time=True,
    )

    if fixed_fpr is not None:
        # A target FPR is provided, we don't require scores of the validation set.
        # W directly use the scores of the test set to compute the threshold. 
        logger.info(
            f"Computing threshold for fixed FPR={fixed_fpr} from test set normal samples"
        )
        normal_scores = s_test[l_test == 0]
        threshold = get_threshold_for_max_rate(normal_scores, max_rate=fixed_fpr)
        num_above_threshold = np.sum(normal_scores > threshold)
        proportion = num_above_threshold / len(normal_scores)
        logger.info(
            f"Chosen threshold {threshold}, leads to {num_above_threshold} "
            f"samples ({proportion:.1%}) above threshold"
        )
    else: 
        # No fixed FPR is provided, we compute the threshold using the validation 
        # set.
        _, s_val = get_scores_generic(
            df=df_val,
            batch_size=4096,
            model=model,
            use_scaler=use_scaler,
            preprocess_fn=preprocess_fn,
            score_fn=get_decision_scores_fn,
        )

        threshold = get_threshold_for_max_rate(s_val=s_val)
        num_above_threshold = np.sum(s_val > threshold)
        proportion = num_above_threshold / len(s_val)
        logger.info(
            f"Chosen threshold {threshold}, leads to {num_above_threshold} "
            f"samples ({proportion:.1%}) above threshold"
        )

    # Here, set preds where attack_technique = "insider" to min(scores) ->
    # It should be classifier as normal to be a false negative.
    if insider_as_fn:
        insider_mask = df_test["attack_technique"].eq("insider")
        if insider_mask.any():
            min_score = np.min(s_test)
            s_test[insider_mask.values] = min_score
            logger.info(
                f"Set {insider_mask.sum()} 'insider' samples to min score ({min_score}) "
                "to be treated as false negatives."
            )

    d_res, preds = get_metrics_treshold(
        labels=l_test,
        scores=s_test,
        model_name=model_name,
        threshold=threshold,
    )

    # Recall per attack
    _df = pd.DataFrame(
        {
            "attack_technique": df_test["attack_technique"],
            "statement_type": df_test["statement_type"],
            "label": l_test,
            "preds": preds,
        }
    )
    recall_per_attack = get_recall_per_attack(df=_df, model_name=model_name)
    d_res.update(recall_per_attack)
    d_res.update(get_balanced_accuracy_per_attack(df=_df, model_name=model_name, recall_per_attack=recall_per_attack))
    d_res.update(get_recall_per_statement_type(df=_df, model_name=model_name))

    training_results.append(d_res)

    return l_test, s_test


def train_lof_gaur(
    df: pd.DataFrame,
    args: argparse.Namespace,
    use_scaler: bool = False,
    use_hybrid: bool = False,
    mode: str = "expert",
):
    init_seeds()  # Foster reproducible results

    model_name = "GAUR and LOF"

    if not use_hybrid:
        model_name += "-no-Li"
    if use_scaler:
        model_name += "-scaler"

    if mode != "expert":
        model_name += f"-{mode}"

    logger.info(f"Training model: {model_name}")
    model = LOF_Gaur(
        n_jobs=n_jobs,
        use_scaler=use_scaler,
        mode=mode,
        use_hybrid=use_hybrid,
    )

    df_train, df_test, df_val = get_traces_per_split(df, use_cache=args.use_cache)

    model.train_model(
        df=df_train,
        model_name=model_name,
    )
    return compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocess_gaur,
        get_decision_scores_fn=decision_score_generic,
        use_scaler=use_scaler,
        fixed_fpr=args.fixed_fpr
    ) + (model_name,)


def train_ocsvm_gaur(
    df: pd.DataFrame,
    args: argparse.Namespace,
    use_scaler: bool = False,
    use_hybrid: bool = False,
    mode: str = "expert",
):
    init_seeds()  # Foster reproducible results

    model_name = "GAUR and OCSVM"

    if not use_hybrid:
        model_name += "-no-Li"

    if use_scaler:
        model_name += "-scaler"

    if mode != "expert":
        model_name += f"-{mode}"

    logger.info(f"Training model: {model_name}")
    model = OCSVM_Gaur(
        nu=0.05,
        kernel="rbf",
        gamma="scale",
        max_iter=10000,
        use_scaler=use_scaler,
        use_hybrid=use_hybrid,
        mode=mode,
    )

    df_train, df_test, df_val = get_traces_per_split(df, use_cache=args.use_cache)

    model.train_model(
        df=df_train,
        model_name=model_name,
    )
    return compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocess_gaur,
        get_decision_scores_fn=decision_score_generic,
        use_scaler=use_scaler,
        fixed_fpr=args.fixed_fpr,
    ) + (model_name,)


def train_ae_gaur(
    df: pd.DataFrame,
    args: argparse.Namespace,
    use_scaler: bool = False,
    use_hybrid: bool = False,
    mode: str = "expert",
):
    init_seeds()  # Foster reproducible results

    model_name = "GAUR and AE"

    if not use_hybrid:
        model_name += "-no-Li"

    if use_scaler:
        model_name += "-scaler"

    if mode != "expert":
        model_name += f"-{mode}"

    logger.info(f"Training model: {model_name}")
    model = AutoEncoder_Gaur(
        device=init_device(),
        learning_rate=0.001,
        epochs=100,
        batch_size=4096,
        # The scaling is performed in the class functions. No need to do anything
        # here besides setting this boolean.
        use_scaler=use_scaler,
        mode=mode,
        use_hybrid=use_hybrid,
        project_paths=config.ppths,
        use_cache=True,
    )

    df_train, df_test, df_val = get_traces_per_split(df, use_cache=args.use_cache)

    model.train_model(df=df_train, model_name=model_name)

    return compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocess_gaur_ae,
        get_decision_scores_fn=decision_score_ae,
        use_scaler=use_scaler,
        fixed_fpr=args.fixed_fpr,
    ) + (model_name,)


def select_models(args: argparse.Namespace) -> dict:
    """Selects a set of model training functions based on the user's input.
    Supports grouping of models and allows selecting all available models.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
        - args.models (List[str]): List of models or model groups to select.
          Special value "all" selects all available models.

    Returns:
        dict: A dictionary mapping model names to their corresponding training functions.
        Each function accepts a DataFrame and a trace type and returns (labels, scores, name).
    """

    AUTHORIZED_GROUPS = {
        # "gaur_no_li": ["ocsvm_no_li", "ae_no_li"],
        "gaur": ["ocsvm", "ae"],
        "ruleid": ["ocsvm_ruleid", "ae_ruleid"],
    }

    MODEL_REGISTRY = {
        # "ocsvm_no_li": lambda df, mode: train_ocsvm_gaur(
        #     df, use_scaler=True, args=args, mode=mode
        # ),
        "ocsvm": lambda df, mode: train_ocsvm_gaur(
            df, use_scaler=True, args=args, use_hybrid=True, mode=mode
        ),
        # "ae_no_li": lambda df, mode: train_ae_gaur(
        #     df, use_scaler=True, args=args, mode=mode
        # ),
        "ae": lambda df, mode: train_ae_gaur(
            df, use_scaler=True, args=args, use_hybrid=True, mode=mode
        ),
        "ocsvm_ruleid": lambda df, mode: train_ocsvm_gaur(
            df, use_scaler=True, args=args, use_hybrid=True, mode="ruleid"
        ),
        "ae_ruleid": lambda df, mode: train_ae_gaur(
            df, use_scaler=True, args=args, use_hybrid=True, mode="ruleid"
        ),
    }

    if "all" in args.models:
        return MODEL_REGISTRY

    expanded = []
    for item in args.models:
        if item in AUTHORIZED_GROUPS:
            expanded.extend(AUTHORIZED_GROUPS[item])
        else:
            expanded.append(item)

    selected = {}
    for name in expanded:
        if name in MODEL_REGISTRY:
            selected[name] = MODEL_REGISTRY[name]
        else:
            logger.warning(f"Unrecognized model '{name}', skipping.")
    return selected


def select_trace_types(args: argparse.Namespace) -> list:
    """Selects valid trace types for training and evaluation based on user input.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
        - args.trace_type (List[str]): List of trace types to use.
          Special value "all" selects all available trace types.

    Returns:
        list: List of valid trace types selected by the user.
    """
    valid_traces = ["expert", "claude", "chatgpt", "llama", "mistral", "gpt-oss"]
    modes = args.trace_type

    if "all" in args.trace_type:
        return ["expert", "claude", "chatgpt", "llama", "mistral", "gpt-oss"]

    for item in modes:
        if item not in valid_traces:
            logger.warning(f"Unrecognized trace type: {item}.")
            modes.remove(item)
    return modes


def train_models(
    df: pd.DataFrame,
    args,
):
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]

    logger.info(
        f"Training - number of attacks {len(df_train[df_train['label'] == 1])}"
        f" and number of normals {len(df_train[df_train['label'] == 0])}"
    )
    logger.info(
        f"Testing - number of attacks {len(df_test[df_test['label'] == 1])}"
        f" and number of normals {len(df_test[df_test['label'] == 0])}"
    )

    # Train models and get their output.
    model_output = {}
    traces = select_trace_types(args)
    models = select_models(args)

    if len(models) == 0:
        logger.critical("No valid model selected, exiting.")
        exit()

    if len(traces) == 0:
        logger.critical("No valid trace type selected, exiting.")

    ruleid_models = {name: fn for name, fn in models.items() if "ruleid" in name}
    used_trace_types = set(traces)
    if ruleid_models:
        used_trace_types.add("expert")

    try:
        for trace in traces:
            # Update where to find socket depending on the trace type.
            gaur_sqld.update_location_mysqlfiles(trace)

            # We don't want to train ruleid models inside the loop. They ignore semantic
            # tags in the trace, we would be training and evaluating the same models
            # multiple time if multiple traces are given.
            for model_name, model_fn in models.items():
                if "ruleid" in model_name:
                    continue  # skip ruleid models here
                res_start = len(training_results)
                inf_start = len(inference_times)
                labels, scores, returned_name = model_fn(df, trace)
                model_output[returned_name] = (labels, scores)

                # Save per-model results (safe for concurrent NFS writers).
                trace_out = config.ppths.trace_type_output_path
                pd.DataFrame(training_results[res_start:]).to_csv(
                    f"{trace_out}results-{model_name}.csv", index=False
                )
                pd.DataFrame(inference_times[inf_start:]).to_csv(
                    f"{trace_out}inference-{model_name}.csv", index=False
                )

        # We train ruleids models outside the loop and they require the expert trace.
        gaur_sqld.update_location_mysqlfiles("expert")

        for model_name, model_fn in ruleid_models.items():
            res_start = len(training_results)
            inf_start = len(inference_times)
            labels, scores, returned_name = model_fn(df, "ruleid")
            model_output[returned_name] = (labels, scores)

            trace_out = config.ppths.trace_type_output_path
            pd.DataFrame(training_results[res_start:]).to_csv(
                f"{trace_out}results-{model_name}.csv", index=False
            )
            pd.DataFrame(inference_times[inf_start:]).to_csv(
                f"{trace_out}inference-{model_name}.csv", index=False
            )

        labels_list = [labels for labels, _ in model_output.values()]
        scores_list = [scores for _, scores in model_output.values()]

        names_list = list(model_output.keys())

        ref_labels = labels_list[0]
        for labels in labels_list[1:]:
            # assert np.array_equal(ref_labels, labels)
            if not np.array_equal(ref_labels, labels):
                logger.critical(f"Label mismatch detected")

        plot_pr_curves_plt_from_scores(
            labels=ref_labels,
            l_scores=scores_list,
            l_model_names=names_list,
        )

        plot_roc_curves_plt_from_scores(
            labels=ref_labels,
            l_scores=scores_list,
            l_model_names=names_list,
        )

        # Save aggregate results across all trace types.
        dfres = pd.DataFrame(training_results)
        filename = f"{config.ppths.output_path}/results.csv"
        dfres.to_csv(filename, index=False)

        dfinf = pd.DataFrame(inference_times)
        fp = config.ppths.inference_filepath + "/inference.csv"
        dfinf.to_csv(fp, index=False)

    finally:
        for tt in used_trace_types:
            stop_server(tt)

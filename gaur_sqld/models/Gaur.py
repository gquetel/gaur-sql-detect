import hashlib
import logging
from pathlib import Path
import pickle
import numpy as np
import os
import pandas as pd
import re
import torch
import torch.nn as nn

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import OneClassSVM


import gaur_sqld.config as config
from .AutoEncoders import AERelu, AESigmoid
from .Li import pre_process_for_li

from gaur_sqld.utils.constants import (
    GAUR_SEM_ACTIONS,
    GAUR_SEM_OBJECTS,
    ProjectPaths,
    mysql_functions,
    mysql_keywords,
)
from gaur_sqld.utils.explain import compute_per_feature_RE

logger = logging.getLogger(__name__)

set_sqlkywds = mysql_functions.union(mysql_keywords)
# We create a pattern with word boundaries to prevent match on substrings.
re_sqlkywds = re.compile(r"\b(?:%s)\b" % "|".join(set_sqlkywds), flags=re.IGNORECASE)

# -------- Feature preprocessing functions --------


def count_sql_keywords(text: str) -> int:
    words = re_sqlkywds.findall(text)
    return len(words)


def parse_trace_chatgpt(trace: str) -> tuple:
    t_tags = {
        "DDL_ALTER": 0,
        "DDL_CREATE": 0,
        "DDL_DROP": 0,
        "DML_DELETE_TRUNCATE": 0,
        "DML_INSERT_REPLACE": 0,
        "DML_MAINTENANCE": 0,
        "DML_SELECT": 0,
        "DML_UPDATE": 0,
        "EXPRESSION_LOGIC": 0,
        "PARTITIONING_STORAGE": 0,
        "PRIVILEGES_SECURITY": 0,
        "PROCEDURAL_LOGIC": 0,
        "REPLICATION_MANAGEMENT": 0,
        "SERVER_ADMIN": 0,
        "SHOW_DESCRIBE_EXPLAIN": 0,
        "STATEMENT_CONTROL": 0,
        "STATEMENT_HELP": 0,
        "STATEMENT_MANAGEMENT": 0,
        "TRANSACTION_CONTROL": 0,
        "WINDOW_ANALYTICS": 0,
    }

    l_ruleid = []
    l_order = []
    l_id = []
    c_sqlkywds = []

    if pd.isna(trace):
        return t_tags, l_ruleid, c_sqlkywds, l_order, l_id

    nodes = trace.split("||-||")[0].split("|")
    for node in nodes:
        if not node:
            continue
        try:
            r_order, symbkind, r_id, r_tag, _, sem_value = node.split(":")
        except ValueError:
            logger.warning(f"Failed to extract all values for the node: {node}. ")
            continue
        # sem_value correspond to literals found in the trace.
        # We count the number of sql keywords in the trace using
        if sem_value:
            sem_value.replace("G_SEMICOLON", ":")
            sem_value.replace("G_PIPE", "|")
            c_sqlkywds.append(count_sql_keywords(sem_value))

        l_ruleid.append(symbkind)
        l_order.append(r_order)
        l_id.append(r_id)
        if r_tag != "":
            t_tags[r_tag] += 1

    return t_tags, l_ruleid, c_sqlkywds, l_order, l_id


def parse_trace_claude(trace: str) -> tuple:
    t_tags = {
        "ADMINISTRATIVE": 0,
        "CLAUSE_COMPONENT": 0,
        "CONSTRAINT_DEFINITION": 0,
        "DATA_IMPORT_EXPORT": 0,
        "DATA_TYPE": 0,
        "DDL_STATEMENT": 0,
        "DML_STATEMENT": 0,
        "ENTRY_POINT": 0,
        "EXPRESSION": 0,
        "FUNCTION_CALL": 0,
        "IDENTIFIER": 0,
        "LITERAL_VALUE": 0,
        "OPTIONAL_MODIFIER": 0,
        "QUERY_STRUCTURE": 0,
        "REPLICATION_CLUSTER": 0,
        "STORED_PROCEDURE": 0,
        "SYNTAX_ELEMENT": 0,
        "TABLE_REFERENCE": 0,
        "TRANSACTION_CONTROL": 0,
        "USER_MANAGEMENT": 0,
    }

    l_ruleid = []
    l_order = []
    l_id = []
    c_sqlkywds = []

    if pd.isna(trace):
        return t_tags, l_ruleid, c_sqlkywds, l_order, l_id

    nodes = trace.split("||-||")[0].split("|")
    for node in nodes:
        if not node:
            continue
        try:
            r_order, symbkind, r_id, r_tag, _, sem_value = node.split(":")
        except ValueError:
            logger.warning(f"Failed to extract all values for the node: {node}. ")
            continue
        # sem_value correspond to literals found in the trace.
        # We count the number of sql keywords in the trace using
        if sem_value:
            sem_value.replace("G_SEMICOLON", ":")
            sem_value.replace("G_PIPE", "|")
            c_sqlkywds.append(count_sql_keywords(sem_value))

        l_ruleid.append(symbkind)
        l_order.append(r_order)
        l_id.append(r_id)
        if r_tag != "":
            t_tags[r_tag] += 1

    return t_tags, l_ruleid, c_sqlkywds, l_order, l_id


def parse_trace_llama(trace: str) -> tuple:
    t_tags = {
        "DCL": 0,
        "DDL": 0,
        "DML": 0,
        "Database": 0,
        "Event": 0,
        "Function": 0,
        "Indexing": 0,
        "Locking": 0,
        "Procedure": 0,
        "Query": 0,
        "Role": 0,
        "Security": 0,
        "Server": 0,
        "Table": 0,
        "Tablespace": 0,
        "Transaction": 0,
        "Trigger": 0,
        "User": 0,
        "Utility": 0,
        "View": 0,
    }

    l_ruleid = []
    l_order = []
    l_id = []
    c_sqlkywds = []

    if pd.isna(trace):
        return t_tags, l_ruleid, c_sqlkywds, l_order, l_id

    nodes = trace.split("||-||")[0].split("|")
    for node in nodes:
        if not node:
            continue
        try:
            r_order, symbkind, r_id, r_tag, _, sem_value = node.split(":")
        except ValueError:
            logger.warning(f"Failed to extract all values for the node: {node}. ")
            continue
        # sem_value correspond to literals found in the trace.
        # We count the number of sql keywords in the trace using
        if sem_value:
            sem_value.replace("G_SEMICOLON", ":")
            sem_value.replace("G_PIPE", "|")
            c_sqlkywds.append(count_sql_keywords(sem_value))

        l_ruleid.append(symbkind)
        l_order.append(r_order)
        l_id.append(r_id)
        if r_tag != "":
            t_tags[r_tag] += 1

    return t_tags, l_ruleid, c_sqlkywds, l_order, l_id


def parse_trace_gpt_oss(trace: str) -> tuple:
    t_tags = {
        "Clause-Modifier": 0,
        "Constraint": 0,
        "DDL-Statement": 0,
        "DML-Statement": 0,
        "Data-Type": 0,
        "Event-Scheduling": 0,
        "Expression": 0,
        "Identifier": 0,
        "Index-Definition": 0,
        "Join-Clause": 0,
        "Literal": 0,
        "Options-List": 0,
        "Partition-Clause": 0,
        "Predicate": 0,
        "Privilege-Control": 0,
        "Replication-Control": 0,
        "Stored-Program": 0,
        "Transaction-Control": 0,
        "Utility-Statement": 0,
        "Window-Function": 0,
    }

    l_ruleid = []
    l_order = []
    l_id = []
    c_sqlkywds = []

    if pd.isna(trace):
        return t_tags, l_ruleid, c_sqlkywds, l_order, l_id

    nodes = trace.split("||-||")[0].split("|")
    for node in nodes:
        if not node:
            continue
        try:
            r_order, symbkind, r_id, r_tag, _, sem_value = node.split(":")
        except ValueError:
            logger.warning(f"Failed to extract all values for the node: {node}. ")
            continue
        # sem_value correspond to literals found in the trace.
        # We count the number of sql keywords in the trace using
        if sem_value:
            sem_value.replace("G_SEMICOLON", ":")
            sem_value.replace("G_PIPE", "|")
            c_sqlkywds.append(count_sql_keywords(sem_value))

        l_ruleid.append(symbkind)
        l_order.append(r_order)
        l_id.append(r_id)
        if r_tag != "":
            t_tags[r_tag] += 1

    return t_tags, l_ruleid, c_sqlkywds, l_order, l_id


def parse_trace_mistral(trace: str) -> tuple:
    t_tags = {
        "Data Definition": 0,
        "Data Import/Export": 0,
        "Data Manipulation": 0,
        "Data Query": 0,
        "Database Management": 0,
        "Locking & Concurrency": 0,
        "Miscellaneous Operations": 0,
        "Replication & Clustering": 0,
        "Resource Management": 0,
        "Security & Privileges": 0,
        "Statement Control": 0,
        "Stored Procedures & Functions": 0,
        "System Information": 0,
        "System Maintenance": 0,
        "System Variables": 0,
        "Temporary Objects": 0,
        "Transaction Control": 0,
        "Triggers & Events": 0,
        "User Management": 0,
        "Views": 0,
    }

    l_ruleid = []
    l_order = []
    l_id = []
    c_sqlkywds = []

    if pd.isna(trace):
        return t_tags, l_ruleid, c_sqlkywds, l_order, l_id

    nodes = trace.split("||-||")[0].split("|")
    for node in nodes:
        if not node:
            continue
        try:
            r_order, symbkind, r_id, r_tag, _, sem_value = node.split(":")
        except ValueError:
            logger.warning(f"Failed to extract all values for the node: {node}. ")
            continue
        # sem_value correspond to literals found in the trace.
        # We count the number of sql keywords in the trace using
        if sem_value:
            sem_value.replace("G_SEMICOLON", ":")
            sem_value.replace("G_PIPE", "|")
            c_sqlkywds.append(count_sql_keywords(sem_value))

        l_ruleid.append(symbkind)
        l_order.append(r_order)
        l_id.append(r_id)
        if r_tag != "":
            t_tags[r_tag] += 1

    return t_tags, l_ruleid, c_sqlkywds, l_order, l_id


def parse_trace_expert(trace: str) -> tuple:
    """
    Parses a GAUR trace and extracts structured information
    about actions, objects, symbol kinds, SQL keyword usage, order, and IDs.

    WARNING: When modifying this function, reset cache: because the caching
    is performed using input DataFrame as identifier, pre_process_for_gaur
    will load a pickle generated using the old preprocessing.

    Args:
        trace (str): A trace string formatted with nodes separated by '|'
                     and fields within each node separated by ':'.
                     Example node: "1:SYM_KIND:ID:ACTION:OBJECT:SEM_VALUE"

    Returns:
        tuple: A 6-element tuple containing:
            - d_action (dict): Counts of each action type (CREATE, DELETE, etc.).
            - d_object (dict): Counts of each object type (TABLE, USER, etc.).
            - l_ruleid (List[str]): List of symbol kinds encountered in the trace.
            - c_sqlkywds (List[int]): Count of SQL keywords found in each sem_value.
            - l_order (List[str]): List of order values from the trace nodes.
            - l_id (List[str]): List of symbol IDs from the trace nodes.
    """

    d_action = {
        "CREATE": 0,
        "DELETE": 0,
        "MODIFY": 0,
        "EXECUTE": 0,
        "READ": 0,
        "A_NONE": 0,
    }

    d_object = {
        "TABLESPACE": 0,
        "TABLE": 0,
        "INDEX": 0,
        "VIEW": 0,
        "USER": 0,
        "PROCEDURE": 0,
        "DATABASE": 0,
        "FUNCTION": 0,
        "INSTANCE": 0,
        "LOGFILE": 0,
        "SERVER": 0,
        "TRIGGER": 0,
        "O_NONE": 0,
    }

    l_ruleid = []
    l_order = []
    l_id = []
    c_sqlkywds = []

    if pd.isna(trace):
        return d_action, d_object, l_ruleid, c_sqlkywds, l_order, l_id

    nodes = trace.split("||-||")[0].split("|")
    for node in nodes:
        if not node:
            continue
        try:
            r_order, symbkind, r_id, r_action, r_object, sem_value = node.split(":")
        except ValueError:
            logger.warning(f"Failed to extract all values for the node: {node}. ")
            continue
        # sem_value correspond to literals found in the trace.
        # We count the number of sql keywords in the trace using
        if sem_value:
            sem_value.replace("G_SEMICOLON", ":")
            sem_value.replace("G_PIPE", "|")
            c_sqlkywds.append(count_sql_keywords(sem_value))

        l_ruleid.append(symbkind)
        l_order.append(r_order)
        l_id.append(r_id)

        d_action[r_action if r_action else "A_NONE"] += 1
        d_object[r_object if r_object else "O_NONE"] += 1

    return d_action, d_object, l_ruleid, c_sqlkywds, l_order, l_id


def _get_gaur_features_from_query(s: pd.Series) -> pd.Series:
    """Extract features from GAUR log entry.

    WARNING: When modifying this function, reset cache: because the caching
    is performed using input DataFrame as identifier, pre_process_for_gaur
    will load a pickle generated using the old preprocessing.

    Args:
        s (pd.Series): _description_

    Returns:
        pd.Series: _description_
    """
    # d_action, d_object, l_ruleid, c_sqlkywds, l_order, l_id
    d_action, d_object, l_ruleid, c_sqlkywds, _, _ = parse_trace_expert(
        s["semantic_tree"]
    )

    # Compute metrics characterizing the count of SQL keywords:
    avg_c_sqlkywds = 0
    max_c_sqlkywds = 0
    min_c_sqlkywds = 0

    if len(c_sqlkywds) > 0:
        avg_c_sqlkywds = sum(c_sqlkywds) / len(c_sqlkywds)
        max_c_sqlkywds = max(c_sqlkywds)
        min_c_sqlkywds = min(c_sqlkywds)

    return pd.Series(
        {
            **s,
            **d_action,
            **d_object,
            "symbkinds": l_ruleid,
            "avg_c_sqlkywds": avg_c_sqlkywds,
            "max_c_sqlkywds": max_c_sqlkywds,
            "min_c_sqlkywds": min_c_sqlkywds,
        }
    )


def _get_gaur_llm_features_from_query(
    s: pd.Series, parse_function: callable
) -> pd.Series:
    """Extract features from GAUR log entry (using chatgpt or claude tags).

    Args:
        s (pd.Series): _description_

    Returns:
        pd.Series: _description_
    """
    d_sem_tag, l_ruleid, c_sqlkywds, _, _ = parse_function(s["semantic_tree"])

    # Compute metrics characterizing the count of SQL keywords:
    avg_c_sqlkywds = 0
    max_c_sqlkywds = 0
    min_c_sqlkywds = 0

    if len(c_sqlkywds) > 0:
        avg_c_sqlkywds = sum(c_sqlkywds) / len(c_sqlkywds)
        max_c_sqlkywds = max(c_sqlkywds)
        min_c_sqlkywds = min(c_sqlkywds)

    return pd.Series(
        {
            **s,
            **d_sem_tag,
            "symbkinds": l_ruleid,
            "avg_c_sqlkywds": avg_c_sqlkywds,
            "max_c_sqlkywds": max_c_sqlkywds,
            "min_c_sqlkywds": min_c_sqlkywds,
        }
    )


def vectorize_symbkinds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorize 'symbkinds' into features, keeping only values between 832 and 1844.

    Args:
        df (pd.DataFrame): DataFrame containing a 'symbkinds' column (list of int values).

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns for the filtered values.
    """
    min_kind_value = 832
    max_kind_value = 1844
    # These values can be found in sql_yacc.cc when building from source.
    # 832 = kind of YYSYMBOL_YYACCEPT
    # 1844 = kind of YYSYMBOL_json_attribute, the maximum value an ruleid can get.

    kind_counts = np.zeros((len(df), max_kind_value + 1), dtype=int)

    for i, kinds in enumerate(df["symbkinds"]):
        if kinds:
            values = np.array(kinds, dtype=int)
            # Filter only values in the specified range
            values = values[(values >= min_kind_value) & (values <= max_kind_value)]
            if len(values) > 0:
                unique_vals, counts = np.unique(values, return_counts=True)
                kind_counts[i, unique_vals] = counts

    kind_df = pd.DataFrame(
        kind_counts[:, min_kind_value : max_kind_value + 1],
        columns=[f"kind_{i}" for i in range(min_kind_value, max_kind_value + 1)],
        index=df.index,
    )

    return pd.concat([df, kind_df], axis=1)


def pre_process_for_gaur(
    df: pd.DataFrame, mode: str, use_cache: bool = False
) -> pd.DataFrame:
    # We use the same cache mechanism to compute features. The preprocessing is
    # not optimized at all so we might as well gain time by saving DataFrame in
    # pickle format.

    str_hash_df = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()
    # TODO: Also include this in datadir if args is set.
    fp_cache = "".join(
        [config.ppths.cache_path, "gaur-features", str_hash_df, ".pkl"]
    )

    if use_cache and os.path.isfile(fp_cache):
        logger.debug(f"Loaded GAUR features pickle located at {fp_cache}")
        return pd.read_pickle(fp_cache)

    match mode:
        case "expert" | "ruleid" | "expert-no-const":
            df_pped = df.apply(_get_gaur_features_from_query, axis=1)

        case "chatgpt":
            df_pped = df.apply(
                lambda row: _get_gaur_llm_features_from_query(
                    row, parse_trace_chatgpt
                ),
                axis=1,
            )
        case "claude":
            df_pped = df.apply(
                lambda row: _get_gaur_llm_features_from_query(
                    row, parse_trace_claude
                ),
                axis=1,
            )
        case "llama":
            df_pped = df.apply(
                lambda row: _get_gaur_llm_features_from_query(
                    row, parse_trace_llama
                ),
                axis=1,
            )
        case "gpt-oss":
            df_pped = df.apply(
                lambda row: _get_gaur_llm_features_from_query(
                    row, parse_trace_gpt_oss
                ),
                axis=1,
            )

        case "mistral":
            df_pped = df.apply(
                lambda row: _get_gaur_llm_features_from_query(
                    row, parse_trace_mistral
                ),
                axis=1,
            )
        case _:
            raise ValueError(f"Unknown mode: {mode}.")
    if use_cache:
        df_pped.to_pickle(fp_cache)

    return df_pped


# -------- Detectors --------


class OCSVM_Gaur:
    def __init__(
        self,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        max_iter: int = -1,
        use_scaler: bool = False,
        use_hybrid: bool = False,
        mode: str = "expert",
        use_cache: bool = False,
    ):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.clf = None
        self.use_cache = use_cache

        self._scaler = StandardScaler()

        self.model_name = None
        self.max_iter = max_iter
        self.use_scaler = use_scaler
        self.use_hybrid = use_hybrid

        # The mode defines which features will be used:
        # - with "expert" we use semantic tags and discard ruleid info
        # - with "ruleid" we map each ruleid to a column (discarding sem data)
        # - with "chatgpt" we use a custom semantic tag preprocessing function
        # - with "claude", we use another preprocessing function.
        self.mode = mode

    def preprocess_for_preds(self, df: pd.DataFrame, drop_og_columns: bool = True):
        df_pped = df.copy()
        labels = np.array(df_pped["label"])
        df_pped = pre_process_for_gaur(df_pped, self.mode, use_cache=self.use_cache)

        # Mode specific post-processing.
        match self.mode:
            case "expert":
                df_pped = df_pped.drop(
                    ["A_NONE", "O_NONE", "query_id", "semantic_tree"], axis=1
                )
            case "ruleid":
                df_pped = df_pped.drop(
                    ["A_NONE", "O_NONE", "query_id", "semantic_tree"], axis=1
                )
                # First remove semantic tags
                # Then expand ruleid into columns.
                df_pped = df_pped.drop(GAUR_SEM_ACTIONS + GAUR_SEM_OBJECTS, axis=1)
                logger.info(f"Vectorizing symbkinds.")
                df_pped = vectorize_symbkinds(df=df_pped)
            case "chatgpt" | "claude" | "llama" | "gpt-oss" | "mistral":
                df_pped = df_pped.drop(["query_id", "semantic_tree"], axis=1)
            case _:
                raise ValueError(f"Unknown mode: {self.mode}.")

        # This field must be removed for all modes.
        df_pped = df_pped.drop(["symbkinds"], axis=1)

        # If we enable hybrid mode, add li features:
        if self.use_hybrid:
            df_pped = pre_process_for_li(df=df_pped)

        if drop_og_columns:
            # Sometimes needed for metrics computation
            df_pped.drop(
                ["label", "full_query"]
                + [
                    "statement_type",
                    "query_template_id",
                    "user_inputs",
                    "attack_id",
                    "attack_technique",
                    "split",
                    "attack_status",
                    "attack_stage",
                    "tamper_method",
                    "template_split",
                ],
                axis=1,
                inplace=True,
                errors="ignore",
            )
        return df_pped, labels

    def preprocess_for_train(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> pd.DataFrame:
        df_pped, _ = self.preprocess_for_preds(
            df=df, drop_og_columns=drop_og_columns
        )
        return df_pped

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
    ):
        self.model_name = model_name
        df_pped = self.preprocess_for_train(df)
        self.feature_columns = df_pped.columns.tolist()
        if self.use_scaler:
            df_pped = self._scaler.fit_transform(df_pped.values)

        model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
            max_iter=self.max_iter,
        )

        model.fit(df_pped)
        self.clf = model


class LOF_Gaur:
    def __init__(
        self,
        n_jobs: int = -1,
        contamination: float = 0.1,
        use_scaler: bool = False,
        use_hybrid: bool = False,
        mode: str = "expert",
        use_cache: bool = False,
    ):
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.use_cache = use_cache

        self._scaler = StandardScaler()

        self.clf = None
        self.model_name = None
        self.use_scaler = use_scaler
        self.use_hybrid = use_hybrid
        self.mode = mode

    def preprocess_for_preds(self, df: pd.DataFrame, drop_og_columns: bool = True):
        df_pped = df.copy()
        labels = np.array(df_pped["label"])
        df_pped = pre_process_for_gaur(df_pped, self.mode, use_cache=self.use_cache)

        # Mode specific post-processing.
        match self.mode:
            case "expert":
                df_pped = df_pped.drop(
                    ["A_NONE", "O_NONE", "query_id", "semantic_tree"], axis=1
                )
            case "ruleid":
                df_pped = df_pped.drop(
                    ["A_NONE", "O_NONE", "query_id", "semantic_tree"], axis=1
                )
                # First remove semantic tags
                # Then expand ruleid into columns.
                df_pped = df_pped.drop(GAUR_SEM_ACTIONS + GAUR_SEM_OBJECTS, axis=1)
                logger.info(f"Vectorizing symbkinds.")
                df_pped = vectorize_symbkinds(df=df_pped)
            case "chatgpt" | "claude" | "llama" | "gpt-oss" | "mistral":
                df_pped = df_pped.drop(["query_id", "semantic_tree"], axis=1)
            case _:
                raise ValueError(f"Unknown mode: {self.mode}.")

        # This field must be removed for all modes.
        df_pped = df_pped.drop(["symbkinds"], axis=1, errors="ignore")

        # If we enable hybrid mode, add li features:
        if self.use_hybrid:
            df_pped = pre_process_for_li(df=df_pped)

        if drop_og_columns:
            df_pped.drop(
                ["label", "full_query"]
                + [
                    "statement_type",
                    "query_template_id",
                    "user_inputs",
                    "attack_id",
                    "attack_technique",
                    "split",
                    "attack_status",
                    "attack_stage",
                    "tamper_method",
                    "template_split",
                ],
                axis=1,
                inplace=True,
                errors="ignore",
            )
        return df_pped, labels

    def preprocess_for_train(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> pd.DataFrame:
        df_pped, _ = self.preprocess_for_preds(
            df=df, drop_og_columns=drop_og_columns
        )
        return df_pped

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
    ):
        self.model_name = model_name
        df_pped = self.preprocess_for_train(df)
        self.feature_columns = df_pped.columns.tolist()

        X_pped = df_pped.values

        if self.use_scaler:
            X_pped = self._scaler.fit_transform(X_pped)

        model = LocalOutlierFactor(n_jobs=self.n_jobs, novelty=True)
        model.fit(X_pped)

        self.clf = model


class AutoEncoder_Gaur:
    def __init__(
        self,
        device,
        learning_rate: float = 0.005,
        epochs: int = 100,
        batch_size: int = 32,
        use_scaler: bool = False,
        use_hybrid: bool = False,
        mode: str = "expert",
        project_paths: ProjectPaths = None,
        use_cache: bool = False,
    ):
        self.clf = None
        self.model_name = None
        self.mode = mode
        self.use_hybrid = use_hybrid
        self.project_paths = project_paths
        self.use_cache = use_cache

        self._scaler = MaxAbsScaler()

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_scaler = use_scaler
        self.device = device

        self.feature_columns = None
        self.mode = mode

    def set_project_paths(self, project_paths: ProjectPaths):
        self.project_paths = project_paths

    def save_model(self):
        if not self.project_paths:
            logger.warning(f"No project_paths provided, skipping model saving.")

        # 0 => This project's output path.
        save_dir = self.project_paths.models_paths[0]
        model_path = os.path.join(save_dir, f"{self.model_name}.pth")
        meta_path = os.path.join(save_dir, f"{self.model_name}_meta.pth")

        torch.save(self.clf.state_dict(), model_path)
        metadata = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "use_scaler": self.use_scaler,
            "use_hybrid": self.use_hybrid,
            "mode": self.mode,
            "feature_columns": self.feature_columns,
            "model_name": self.model_name,
            "scaler": self._scaler if self.use_scaler else None,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved model to: {model_path} and metadata to: {meta_path}")

    def load_model(self, model_name: str):
        self.model_name = model_name
        if not self.project_paths:
            raise ValueError(
                "No project_paths provided, cannot determine model location."
            )
        models_paths = self.project_paths.models_paths
        for save_dir in models_paths:
            model_path = os.path.join(save_dir, f"{self.model_name}.pth")
            meta_path = os.path.join(save_dir, f"{self.model_name}_meta.pth")

            if Path(model_path).exists() and Path(meta_path).exists():
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)

                # A dict is returned
                self.learning_rate = metadata.get("learning_rate")
                self.epochs = metadata.get("epochs")
                self.batch_size = metadata.get("batch_size")
                self.use_scaler = metadata.get("use_scaler")
                self.use_hybrid = metadata.get("use_hybrid")
                self.mode = metadata.get("mode")
                self.feature_columns = metadata.get("feature_columns")
                if self.use_scaler:
                    self._scaler = metadata.get("scaler")

                # Now load model
                input_dim = len(self.feature_columns)
                if self.use_scaler:
                    self.clf = AESigmoid(input_dim=input_dim)
                else:
                    self.clf = AERelu(input_dim=input_dim)
                self.clf.to(self.device)
                self.clf.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.clf.eval()
                logger.info(f"Loaded model '{self.model_name}' from {model_path}")
                return
        raise FileNotFoundError(
            f"Model '{self.model_name}' not found in any of the search paths:\n"
            + "\n".join(f"  - {path}" for path in models_paths)
        )

    def preprocess_for_preds(self, df: pd.DataFrame, drop_og_columns: bool = True):
        df_pped = df.copy()
        labels = np.array(df_pped["label"])
        df_pped = pre_process_for_gaur(df_pped, self.mode, use_cache=self.use_cache)

        # Mode specific post-processing.
        match self.mode:
            case "expert":
                df_pped = df_pped.drop(
                    ["A_NONE", "O_NONE", "query_id", "semantic_tree"], axis=1
                )

            case "expert-no-const":
                df_pped = df_pped.drop(
                    ["A_NONE", "O_NONE", "query_id", "semantic_tree"]
                    + [
                        "TABLESPACE",
                        "TRIGGER",
                        "USER",
                        "VIEW",
                        "SERVER",
                        "INSTANCE",
                        "LOGFILE",
                    ],
                    axis=1,
                )

            case "ruleid":
                df_pped = df_pped.drop(
                    ["A_NONE", "O_NONE", "query_id", "semantic_tree"], axis=1
                )
                # First remove semantic tags
                # Then expand ruleid into columns.
                df_pped = df_pped.drop(GAUR_SEM_ACTIONS + GAUR_SEM_OBJECTS, axis=1)
                logger.info(f"Vectorizing symbkinds.")
                df_pped = vectorize_symbkinds(df=df_pped)
            case "chatgpt" | "claude" | "llama" | "gpt-oss" | "mistral":
                df_pped = df_pped.drop(["query_id", "semantic_tree"], axis=1)
            case _:
                raise ValueError(f"Unknown mode: {self.mode}.")

        # This field must be removed for all modes.
        df_pped = df_pped.drop(["symbkinds"], axis=1)

        # If we enable hybrid mode, add li features:
        if self.use_hybrid:
            df_pped = pre_process_for_li(df=df_pped)

        if drop_og_columns:
            # Sometimes needed for metrics computation
            df_pped.drop(
                ["label", "full_query"]
                + [
                    "statement_type",
                    "query_template_id",
                    "user_inputs",
                    "attack_id",
                    "attack_technique",
                    "split",
                    "attack_status",
                    "attack_stage",
                    "tamper_method",
                    "template_split",
                ],
                axis=1,
                inplace=True,
                errors="ignore",
            )
        return df_pped, labels

    def _dataframe_to_tensor_batched(self, df, batch_size=4096):
        """
        Used during testing.

        Args:
            df (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 4096.

        Returns:
            _type_: _description_
        """
        n_samples = len(df)
        tensors = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            df_batch = df.iloc[i:batch_end]
            batch_dense = df_batch.values
            if self.use_scaler:
                batch_dense = self._scaler.transform(batch_dense)
            tensors.append(torch.FloatTensor(batch_dense))

        return torch.cat(tensors, dim=0).to(self.device)

    def preprocess_for_train(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> pd.DataFrame:
        """Preprocess data for training. We ignore label data.
        Args:
            df (pd.DataFrame): Input dataframe
            drop_og_columns (bool, optional): Whether to drop original columns. Defaults to True.
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df_pped, _ = self.preprocess_for_preds(
            df=df, drop_og_columns=drop_og_columns
        )
        return df_pped

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
    ):
        # Init variables for training + model
        self.model_name = model_name
        df_pped = self.preprocess_for_train(df)

        self.feature_columns = df_pped.columns.tolist()
        input_dim = len(self.feature_columns)

        # Let's apply Scaler here and not in preprocess, as we want to keep
        # information about the columns
        df_pped = np.array(df_pped)

        # If scaling =>
        if self.use_scaler:
            scaled_data = self._scaler.fit_transform(df_pped)
            train_data = torch.FloatTensor(scaled_data)
            self.clf = AESigmoid(
                input_dim=input_dim,
            )
        else:
            train_data = torch.FloatTensor(df_pped)
            self.clf = AERelu(input_dim=input_dim)
        self.clf.to(self.device)

        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.learning_rate)

        self.clf.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i : i + self.batch_size]
                batch = batch.to(self.device)

                optimizer.zero_grad()
                reconstructed = self.clf(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.debug(
                f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(train_data):.6f}"
            )
        self.save_model()

        # Training is done, let's evaluate reconstruction error:
        compute_per_feature_RE(
            model_name=self.model_name,
            clf=self.clf,
            tensor_data=train_data,
            batch_size=self.batch_size,
            columns_name=self.feature_columns,
            device=self.device,
        )

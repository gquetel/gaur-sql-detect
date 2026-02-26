import sys
import os
import pandas as pd
import numpy as np
import random
import time

# Required to locate xpgaur correctly.
project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(project_root)

import gaur_sqld

import gaur_sqld.config as config
from constants import AutoEncoder_Li, AutoEncoder_SecureBERT
from gaur_sqld.utils.trainers import init_device
from gaur_sqld.models.Gaur import AutoEncoder_Gaur
from gaur_sqld.utils.traces_collector import get_traces_from_df

from wafamole.models.model import Model
from wafamole.evasion.evasion import EvasionEngine
from wafamole.payloadfuzzer.sqlfuzzer import SqlFuzzer, SyntPreservingSqlFuzzer


# Some wrapper around SecureBERT + AE
# WAFAMOLE Performs mutation on the payload, the model is trained on the full query
# This wrapper reconstruct full queries for detection given WAFAMOLE mutations.
class WafamoleSecureBERT(Model):
    def __init__(self, clf: AutoEncoder_SecureBERT):
        super().__init__()
        self.clf = clf

    def set_template(self, template: str):
        self.template = template

    def extract_features(self, value: str):
        full_query = self.template.format(value=value)
        df = pd.DataFrame({"full_query": [full_query], "label": [1]})
        embeddings, _ = self.clf.preprocess_for_preds(df=df)
        return embeddings

    def classify(self, value: str):
        embedding = self.extract_features(value)
        confidence = -(self.clf.clf.decision_function(embedding, is_tensor=False))
        return confidence


class WafamoleGAURAE(Model):
    def __init__(self, clf: AutoEncoder_Gaur):
        super().__init__()
        self.clf = clf
        self.template = None

    def set_template(self, template: str):
        self.template = template

    def extract_features(self, value: str):
        full_query = self.template.format(value=value)
        df = pd.DataFrame({"full_query": [full_query], "label": [1]})
        traces = get_traces_from_df(df=df, use_cache=False, disable_tqdm=True)
        traces = pd.concat([traces, df], axis=1)
        X_gaur, _ = self.clf.preprocess_for_preds(df=traces)
        X_scaled = self.clf._scaler.transform(X_gaur.values)
        return X_scaled

    def classify(self, value: str):
        X = self.extract_features(value)
        scores = -(self.clf.clf.decision_function(X))
        return scores


class WafamoleLiAE(Model):
    def __init__(self, clf: AutoEncoder_Li):
        super().__init__()
        self.clf = clf
        self.template = None

    def set_template(self, template: str):
        self.template = template

    def extract_features(self, value: str):
        full_query = self.template.format(value=value)
        df = pd.DataFrame({"full_query": [full_query], "label": [1]})
        X_li, _ = self.clf.preprocess_for_preds(df=df)
        X_li.drop(["full_query", "label"], axis=1, inplace=True)
        X_scaled = self.clf._scaler.transform(X_li.values)
        return X_scaled

    def classify(self, value: str):
        X = self.extract_features(value)
        scores = -(self.clf.clf.decision_function(X))
        return scores


def load_AutoEncoder_Li_model() -> AutoEncoder_Li:
    model_name = "Li and AE-scaler"
    model = AutoEncoder_Li(
        device=init_device(),
        project_paths=gaur_sqld.ppths,
        learning_rate=0.005,
        epochs=100,
        batch_size=8192,
        use_scaler=True,
    )
    model.load_model(model_name)
    return model


def load_secureBERT_ae_model() -> AutoEncoder_SecureBERT:
    model_name = "SecureBERT and AE"
    model = AutoEncoder_SecureBERT(
        device=init_device(),
        project_paths=gaur_sqld.ppths,
        learning_rate=0.001,
        epochs=100,
        batch_size=512,
    )
    model.load_model(model_name)
    return model


def load_AutoEncoder_Gaur_model(trace_type: str = "mistral") -> AutoEncoder_Gaur:
    model = AutoEncoder_Gaur(
        device=init_device(),
        learning_rate=0.001,
        epochs=100,
        batch_size=4096,
        use_scaler=True,
        mode=trace_type,
        use_hybrid=True,
        project_paths=config.ppths,
        use_cache=False,
    )
    model.load_model("GAUR and AE-scaler-mistral")
    return model


def evaluate_models_on_queries(
    models: dict,
    tandp: tuple[list, list],
    max_rounds: int = 100,
    round_size: int = 20,
    timeout: int = 60,
    fuzzer: SqlFuzzer | SyntPreservingSqlFuzzer = SqlFuzzer,
):
    results = []

    for template, payload in tandp:
        # All thresholds obtained from training.
        # Ultimately we want to save them in the metadata.

        # TODO: Make sure that both model classify the samples as attack first.
        print(f"Template: {template}")
        print(f"Payload: {payload}")
        print(f"Full query: {template.format(value=payload)}")

        entry = {"template": template, "payload": payload}

        for name, (wrapper, threshold) in models.items():
            wrapper.set_template(template)
            print(f"WAF-A-MoLE {name.upper()}")

            engine = EvasionEngine(model=wrapper, fuzzer=fuzzer)
            try:
                conf, ev_payload, rounds = engine.evaluate(
                    payload=payload,
                    max_rounds=max_rounds,
                    round_size=round_size,
                    timeout=timeout,
                    threshold=threshold,
                )
            except TimeoutError:
                # Sometimes the Timme
                # Unknown conf, data is not relevant anyway
                conf = [100]
                ev_payload = "Unknown"
                # We mimic reaching a maximum round budget while we
                # reached a maximum time budget.
                rounds = max_rounds

            entry[f"{name}_conf"] = conf[0]
            entry[f"{name}_payload"] = ev_payload
            entry[f"{name}_rounds"] = rounds
            entry[f"{name}_status"] = conf[0] < threshold
        results.append(entry)

    return results


def extract_template_payload(row):
    fq = row["full_query"]
    # Guess I messed up the saving of the user inputs.
    ui = row["user_inputs"].strip()
    idx = fq.find(ui)

    if idx == -1:
        return None

    template = fq[:idx] + "{value}" + fq[idx + len(ui) :]
    return template, ui


def get_templates_and_payloads(n: int = 100):
    dataset_path = "../../data/dataset.csv"
    # We restrict our selection on template with only a single user inputs
    # In the dataset we didn't provide separators between the different user inputs,
    # which means that we don't  know what part goes where on templates with multiple inputs.
    valid_templates = [
        "airport-D1",
        "airport-D2",
        "airport-D4",
        "airport-D5",
        "airport-D6",
        "airport-D7",
        "airport-D8",
        "airport-S1,",
        "airport-S3",
        "airport-S5",
        "airport-S6",
        "airport-S7",
        "airport-S9",
        "airport-S12",
        "airport-S14",
        "airport-S15",
        "airport-S16",
        "airport-S21",
        "airport-S23",
    ]
    dataset = pd.read_csv(dataset_path)

    # Randomly select templates
    filtered = dataset[
        (dataset["query_template_id"].isin(valid_templates))
        & (dataset["label"] == 1)
    ]
    shuffled_rows = filtered.sample(frac=1, random_state=2)
    results = []

    for _, row in shuffled_rows.iterrows():
        res = extract_template_payload(row)
        if res is not None:
            results.append(res)
            if len(results) == n:
                return results
    raise ValueError(
        f"Not enough items in dataset to reach desired number of templates."
    )


if __name__ == "__main__":
    start_time = time.time()  # Start timing
    # This is where my secureBERT and Li models were saved.
    other_models_path = "/home/infres/gquetel/repos/sqlia-dataset/models/output/models/"
    config.ppths.add_model_path(other_models_path)

    # Configure GAUR trace mode, we use Mistral because this is one of the model for which
    # we showed the semantic tags in a table (they are short). If we want to provide
    # a representation of the features in the paper, this will be more coherent
    # than displaying features out of nowhere.
    trace_type = "mistral"
    gaur_sqld.update_location_mysqlfiles(trace_type)

    # Get  random templates of queries and their payloads.
    tandp = get_templates_and_payloads(100)

    # WAF-A-MoLE works as follows:
    # At each timestep, we compute ROUND_SIZE mutants using the allowed mutators.
    # Once done, they are sorted based on the confidence score and the one with the lowest score
    # is used to generate ROUND_SIZE more mutants on the next round.
    # We repeast until the score is below the threshold or we performed MAX_ROUNDS rounds.
    MAX_ROUNDS = 1000
    ROUND_SIZE = 20
    TIMEOUT = 300

    all_runs = []

    models = {
        "gaur": (
            WafamoleGAURAE(load_AutoEncoder_Gaur_model()),
            0.01904393918812275,
        ),
        "sbert": (
            WafamoleSecureBERT(load_secureBERT_ae_model()),
            3.0330324989336077e-06,
        ),
        # "li": (
        #     WafamoleLiAE(load_AutoEncoder_Li_model()),
        #     0.0315113440155983,
        # ),
    }

    random.seed(2)
    np.random.seed(2)

    # print("Evaluating using SyntPreservingSqlFuzzer...")
    # results_syn = evaluate_models_on_queries(
    #     models,
    #     tandp,
    #     max_rounds=MAX_ROUNDS,
    #     round_size=ROUND_SIZE,
    #     timeout=TIMEOUT,
    #     fuzzer=SyntPreservingSqlFuzzer,
    # )
    # df_syn = pd.DataFrame(results_syn)

    print("Evaluating using full SqlFuzzer...")
    results_full = evaluate_models_on_queries(
        models,
        tandp,
        max_rounds=MAX_ROUNDS,
        round_size=ROUND_SIZE,
        timeout=TIMEOUT,
        fuzzer=SqlFuzzer,
    )
    df_full = pd.DataFrame(results_full)

    # Combine run for global stats later
    # df_syn["fuzzer"] = "synt"
    df_full["fuzzer"] = "full"

    # df_all = pd.concat([df_syn, df_full], ignore_index=True)
    df_all = df_full

    df_all.to_csv("raw_results.csv", index=False)

    summary = (
        df_all.assign(
            gaur_rounds_success=df_all["gaur_rounds"].where(df_all["gaur_status"]),
            sbert_rounds_success=df_all["sbert_rounds"].where(
                df_all["sbert_status"]
            ),
        )
        .groupby(["fuzzer"])
        .agg(
            gaur_success_count=("gaur_status", "sum"),
            sbert_success_count=("sbert_status", "sum"),
            gaur_avg_rounds=("gaur_rounds_success", "mean"),
            sbert_avg_rounds=("sbert_rounds_success", "mean"),
        )
    )
    summary["samples_count"] = len(tandp)
    summary.to_csv("summary.csv")
    end_time = time.time()
    elapsed = end_time - start_time
    print(
        f"Experiment completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)."
    )

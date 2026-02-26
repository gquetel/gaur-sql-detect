import math
from pathlib import Path
from typing import Literal
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import logging

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc,
)

import gaur_sqld.config as config

logger = logging.getLogger(__name__)

def get_ci(score, n):
    """ 
    From: https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
    
    Args:
        score (_type_): _description_
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
    z = 1.96  # 95%
    interval = z * math.sqrt((score * (1 - score)) / n)
    return interval

def get_metrics_treshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    model_name: str = "",
):

    preds = (scores >= threshold).astype(int)

    accuracy = f"{accuracy_score(labels, preds)* 100:.2f}%"
    f1 = f"{f1_score(labels, preds)* 100:.2f}%"
    precision = f"{precision_score(labels, preds) * 100:.2f}%"
    recall = f"{recall_score(labels, preds) * 100:.2f}%"

    p, r, _ = precision_recall_curve(labels, scores, pos_label=1)
    auprc = auc(r, p)

    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)

    C = confusion_matrix(labels, preds, labels=[0, 1])
    TN, FP, _, _ = C.ravel()
    FPR = FP / (FP + TN)
    achieved_fpr = f"{FPR* 100:.5f}%"

    auroc_ci = get_ci(auroc, len(scores))
    auprc_ci = get_ci(auprc, len(scores))

    logger.info(f"Metrics for {model_name}.")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"False Positive Rate: {achieved_fpr}")

    logger.info(f"ROC-AUC: {auroc:.4f}, CI {auroc_ci:.4f}")
    logger.info(f"AUPRC: {auprc:.4f}, CI {auprc_ci:.4f}")

    return (
        {
            "model": model_name,
            "fone": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "fpr": achieved_fpr,
            "auprc": f"{auprc:.4f}",
            "rocauc": f"{auroc:.4f}",
            "auroc_ci": f"{auroc_ci:.4f}",
            "auprc_ci": f"{auprc_ci:.4f}",
        },
        preds,
    )


def print_and_save_metrics(
    all_targets: list,
    all_predictions: list,
    all_probas: np.ndarray,
    average: Literal["binary", "macro", "micro"] = "binary",
    model: str = "",
    silent: bool = False,
):
    """Print the metrics for the given targets and predictions
    Metrics are: Accuracy, F1 Score, Precision, Recall

    Args:
        all_targets (list): List of targets
        all_predictions (list): List of predictions
    """
    accuracy = f"{accuracy_score(all_targets, all_predictions)* 100:.2f}%"
    f1 = f"{f1_score(all_targets, all_predictions, average=average)* 100:.2f}%"
    sprecision = (
        f"{precision_score(all_targets, all_predictions, average=average)* 100:.2f}%"
    )
    srecall = (
        f"{recall_score(all_targets, all_predictions, average=average)* 100:.2f}%"
    )

    # Addind labels information prevent from only returning a single value
    # causing the not enough values to unpack error.
    C = confusion_matrix(all_targets, all_predictions, labels=[0, 1])
    TN, FP, _, _ = C.ravel()
    FPR = FP / (FP + TN)
    sfpr = f"{FPR* 100:.2f}%"

    precision, recall, _ = precision_recall_curve(all_targets, all_probas[:, 1])
    auc_pr = auc(recall, precision)

    fpr, tpr, _ = roc_curve(all_targets, all_probas[:, 1])
    auc_roc = auc(fpr, tpr)

    if not silent:
        logger.info(f"Metrics for {model}, using {average} average.")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"F1 Score: {f1}")
        logger.info(f"Precision: {sprecision}")
        logger.info(f"Recall: {srecall}")
        logger.info(f"False Positive Rate: {sfpr}")
        logger.info(f"AUPRC : {auc_pr:.4f}")
        logger.info(f"ROC-AUC: {auc_roc:.4f}")

    return {
        "model": model,
        "fone": f1,
        "accuracy": accuracy,
        "precision": sprecision,
        "recall": srecall,
        "fpr": sfpr,
        "auprc": f"{auc_pr:.4f}",
        "rocauc": f"{auc_roc:.4f}",
    }




def get_recall_per_attack(df: pd.DataFrame, model_name: str, suffix: str = ""):
    """Display Recall score per technique from a dataframe with preds."""
    techniques = df.loc[df["label"] == 1, "attack_technique"].unique().tolist()
    logger.info(f"Computing recall for model: {model_name}{suffix}")

    d_res = {}

    for i, technique in enumerate(techniques):
        mask = df["attack_technique"] == technique
        preds = df.loc[mask, "preds"]
        labels = df.loc[mask, "label"]
        srecall = f"{recall_score(labels, preds, average='binary')* 100:.2f}%"
        logger.info(f"Recall for technique {technique}: {srecall}")
        d_res[f"recall{technique}"] = srecall

    return d_res


def get_balanced_accuracy_per_attack(
    df: pd.DataFrame, model_name: str, recall_per_attack: dict
):
    """Compute multiclass-style balanced accuracy treating each attack technique as a separate class.

    Balanced accuracy = mean recall across all classes (normal + each attack technique).
    Reuses the already-computed per-technique recalls from get_recall_per_attack().
    """
    class_recalls = []

    # TNR (recall for normal class)
    normal_mask = df["label"] == 0
    if normal_mask.any():
        normal_preds = df.loc[normal_mask, "preds"]
        tnr = (normal_preds == 0).sum() / len(normal_preds)
        class_recalls.append(tnr)

    # Per-technique recalls
    for key, value in recall_per_attack.items():
        tech_recall = float(value.strip("%")) / 100.0
        class_recalls.append(tech_recall)

    bal_acc = np.mean(class_recalls) if class_recalls else 0.0

    result = f"{bal_acc * 100:.2f}%"
    logger.info(f"Balanced Accuracy Per Technique for {model_name}: {result}")
    return {
        "balanced_accuracy_per_technique": result,
    }


def get_recall_per_statement_type(df: pd.DataFrame, model_name: str, suffix: str = ""):
    """Display Recall score per statement type from a dataframe with preds."""
    statement_types = df.loc[df["label"] == 1, "statement_type"].unique().tolist()
    logger.info(f"Computing recall per statement type for model: {model_name}{suffix}")

    d_res = {}

    for statement_type in statement_types:
        mask = df["statement_type"] == statement_type
        preds = df.loc[mask, "preds"]
        labels = df.loc[mask, "label"]
        srecall = f"{recall_score(labels, preds, average='binary')* 100:.2f}%"
        logger.info(f"Recall for statement type {statement_type}: {srecall}")
        d_res[f"recall_{statement_type}"] = srecall

    return d_res

def plot_pr_curves_plt_from_scores(
    labels, l_scores: list, l_model_names: list, suffix: str = ""
):
    fig, ax = plt.subplots(figsize=(12, 10))
    folder_name = f"{config.ppths.output_path}pr_curves/"
    Path(folder_name).mkdir(exist_ok=True, parents=True)

    for scores, model_name in zip(l_scores, l_model_names):
        assert isinstance(scores, np.ndarray)
        precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
        auprc = auc(recall, precision)

        # Plot the curve
        ax.plot(recall, precision, label=f"{model_name} ({auprc:.4f})")

        # Also let's save the results:
        filepath = folder_name + f"{model_name}{suffix}.csv"
        pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
            filepath, index=False
        )

    # y = prevalence
    x = [0, 1]
    y = [sum(labels) / len(labels)] * len(x)
    ax.plot(x, y, "k--", alpha=0.6, label=f"Random Classifier = {y[0]:.4f}")

    # Customize plot
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("AUPRC Comparison")
    ax.legend()

    ax.grid(True, alpha=0.3)

    # plt.tight_layout()
    plt.savefig(f"{folder_name}auprc_curves{suffix}.png")


def plot_roc_curves_plt_from_scores(
    labels: list, l_scores: list, l_model_names: list, suffix: str = ""
):
    fig, ax = plt.subplots(figsize=(12, 10))
    folder_name = f"{config.ppths.output_path}roc_curves/"
    Path(folder_name).mkdir(exist_ok=True, parents=True)

    for scores, model_name in zip(l_scores, l_model_names):
        # I lost myself with all of the type conversion for the different pipelines
        # Let's just impose a numpy array:
        assert isinstance(scores, np.ndarray)

        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{model_name} ({auroc:.4f})")

        # Also let's save the results:
        filepath = folder_name + f"{model_name}{suffix}.csv"
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(filepath, index=False)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Classifier")
    ax.legend()
    ax.set_title("ROC Curves Comparison")
    ax.grid(True, alpha=0.3)

    plt.savefig(f"{folder_name}roc_curves{suffix}.png")


def compute_per_feature_RE(
    model_name: str, tensor_data, clf, columns_name: list, batch_size: int, device
):
    clf.eval()
    total_squared_error = torch.zeros(tensor_data.shape[1], device=device)
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(tensor_data), batch_size):
            batch = tensor_data[i : i + batch_size].to(device)
            recons = clf(batch)

            errors = (recons - batch) ** 2  # shape: (batch_size, D)
            batch_sum = errors.sum(dim=0)  # sum of squared errors per feature
            total_squared_error += batch_sum
            total_samples += batch.size(0)

    mean_errors = (total_squared_error / total_samples).cpu().numpy()
    per_feature_error = pd.Series(mean_errors, index=columns_name, name="MSE")

    folder_name = f"{config.ppths.output_path}per_feature_mse/"
    Path(folder_name).mkdir(exist_ok=True, parents=True)
    save_path = Path(folder_name) / f"{model_name}_per_feature_mse.csv"
    per_feature_error.to_csv(save_path, index=True)

    logger.debug(f"Saved MSE per feature under: {save_path}.")
    return per_feature_error

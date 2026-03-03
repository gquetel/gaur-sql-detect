import csv

CSV_PATH = "/home/gquetel/experiements-results/2026-03-03/GAUR/results-merged.csv"

GAUR_MODEL_MAP = {
    "GAUR and OCSVM-scaler-expert":   ("OCSVM", "expert"),
    "GAUR and OCSVM-scaler-claude":   ("OCSVM", "claude"),
    "GAUR and OCSVM-scaler-chatgpt":  ("OCSVM", "chatgpt"),
    "GAUR and OCSVM-scaler-llama":    ("OCSVM", "llama"),
    "GAUR and OCSVM-scaler-mistral":  ("OCSVM", "mistral"),
    "GAUR and OCSVM-scaler-gpt-oss":  ("OCSVM", "gptoss"),
    "GAUR and OCSVM-scaler-ruleid":   ("OCSVM", "ruleid"),
    "GAUR and AE-scaler-expert":      ("AE",    "expert"),
    "GAUR and AE-scaler-claude":      ("AE",    "claude"),
    "GAUR and AE-scaler-chatgpt":     ("AE",    "chatgpt"),
    "GAUR and AE-scaler-llama":       ("AE",    "llama"),
    "GAUR and AE-scaler-mistral":     ("AE",    "mistral"),
    "GAUR and AE-scaler-gpt-oss":     ("AE",    "gptoss"),
    "GAUR and AE-scaler-ruleid":      ("AE",    "ruleid"),
}

BASELINE_MODEL_MAP = {
    "Li and OCSVM-scaler":  ("li",         "OCSVM"),
    "Li and AE-scaler":     ("li",         "AE"),
    "SecureBERT and OCSVM": ("securebert", "OCSVM"),
    "SecureBERT and AE":    ("securebert", "AE"),
}

GAUR_SEMANTIC_LABELS = {
    "expert":  r"\gexpert[]",
    "claude":  r"\gclaude[]",
    "chatgpt": r"\gchatgpt[]",
    "llama":   r"\gllama[]",
    "mistral": r"\gmistral[]",
    "gptoss":  r"\ggptoss[]",
    "ruleid":  r"\gruleid[]",
}

BASELINE_LABELS = {
    "li":         r"\li[]",
    "securebert": r"\securebert[]",
}

RECALL_COLS = [
    "recallboolean", "recallerror", "recallunion",
    "recallstacked", "recalltime", "recallinline", "recallinsider",
]

METRIC_COLS = ["auprc", "auroc", "f1"] + RECALL_COLS


def parse_pct(s):
    return float(s.strip().rstrip("%"))

def parse_val(s):
    return float(s.strip())

def extract_row(row):
    return {
        "f1":       parse_pct(row["fone"]),
        "auprc":    parse_val(row["auprc"]) * 100,
        "auroc":    parse_val(row["rocauc"]) * 100,
        "auprc_ci": parse_val(row["auprc_ci"]) * 100,
        "auroc_ci": parse_val(row["auroc_ci"]) * 100,
        **{col: parse_pct(row.get(col, "0.00%")) for col in RECALL_COLS},
    }

def load_data(path):
    gaur_rows = {}
    baseline_rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["model"].strip()
            if key in GAUR_MODEL_MAP:
                clf, sem = GAUR_MODEL_MAP[key]
                gaur_rows[(clf, sem)] = extract_row(row)
            elif key in BASELINE_MODEL_MAP:
                label_key, clf = BASELINE_MODEL_MAP[key]
                baseline_rows[(label_key, clf)] = extract_row(row)
    return gaur_rows, baseline_rows

def fmt(v, bold=False):
    s = f"{v:.2f}"
    return f"\\textbf{{{s}}}" if bold else s

def make_row_pair(model_label, clf, d, bold_set):
    """Return two LaTeX row strings for a single model entry."""
    label_latex = rf"\shortstack[l]{{{model_label} \\ and {clf}}}"
    recall_cells = " & ".join(
        rf"\multirow{{2}}{{*}}{{{fmt(d[col], col in bold_set)}}}"
        for col in RECALL_COLS
    )
    row1 = (
        f"      \\multirow{{2}}{{*}}{{{label_latex}}}"
        f" & {fmt(d['auprc'], 'auprc' in bold_set)}"
        f" & {fmt(d['auroc'], 'auroc' in bold_set)}"
        f" & \\multirow{{2}}{{*}}{{{fmt(d['f1'], 'f1' in bold_set)}}}"
        f" & {recall_cells} \\\\"
    )
    empties = " & ".join([""] * (len(RECALL_COLS) + 1))
    row2 = (
        f"      & $\\pm${d['auprc_ci']:.2f}"
        f" & $\\pm${d['auroc_ci']:.2f}"
        f" & {empties} \\\\"
    )
    return row1, row2

def generate_table(gaur_rows, baseline_rows):
    # Top 3 GAUR pipelines by F1 (descending)
    top3 = sorted(gaur_rows.items(), key=lambda x: x[1]["f1"], reverse=True)[:3]

    baseline_order = [
        ("li",         "OCSVM"),
        ("li",         "AE"),
        ("securebert", "OCSVM"),
        ("securebert", "AE"),
    ]

    # Collect all displayed rows to compute per-column best values
    all_rows = [d for _, d in top3] + [
        baseline_rows[k] for k in baseline_order if k in baseline_rows
    ]
    best = {col: max(d[col] for d in all_rows) for col in METRIC_COLS}

    def bold_set(d):
        return {col for col in METRIC_COLS if abs(d[col] - best[col]) < 1e-6}

    lines = [
        r"\begin{table}",
        r"  \centering",
        r"  \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{l|ccc|ccccccc}",
        r"      \hline",
        r"      \multirow{2}{*}{{\textbf{Model}}} &",
        r"      \multirow{2}{*}{{\textbf{AUPRC}}} &",
        r"      \multirow{2}{*}{{\textbf{AUROC}}} &",
        r"      \multirow{2}{*}{{\textbf{F1}}}    &",
        r"      \multicolumn{7}{c}{\textbf{Recall per attack technique (\%)}} \\",
        r"",
        r"      & & & & Boolean & Error & Union & Stacked & Time & Inline & Insider \\",
        r"      \hline",
    ]

    for (clf, sem), d in top3:
        r1, r2 = make_row_pair(GAUR_SEMANTIC_LABELS[sem], clf, d, bold_set(d))
        lines += [r1, r2, r"      \hline"]

    lines.append(r"      \hline")  # double \hline between GAUR and baselines

    for label_key, clf in baseline_order:
        if (label_key, clf) not in baseline_rows:
            continue
        d = baseline_rows[(label_key, clf)]
        r1, r2 = make_row_pair(BASELINE_LABELS[label_key], clf, d, bold_set(d))
        lines += [r1, r2, r"      \hline"]

    lines += [
        r"    \end{tabular}",
        r"  }",
        r"  \caption{Performance comparison across feature extraction and anomaly detection approaches."
        r" All metrics are in percentages (\%). Best results per metric are shown in bold.}",
        r"  \label{tab:perfs-vs-sota}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    gaur_rows, baseline_rows = load_data(CSV_PATH)
    print(generate_table(gaur_rows, baseline_rows))

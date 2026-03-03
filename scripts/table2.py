import csv                     

CSV_PATH = "/home/gquetel/experiements-results/2026-03-03/GAUR/results-merged.csv"                                                                                                                                 

# Map CSV model names to (classifier, semantic_model)                                                                                                                                                              
MODEL_MAP = {   
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

SEMANTIC_MODELS = ["expert", "chatgpt", "claude", "llama", "mistral", "gptoss", "ruleid"]
CLASSIFIERS    = ["OCSVM", "AE"]

LATEX_LABELS = {
    "expert":  r"\expert[]",
    "claude":  r"\claude[]",
    "chatgpt": r"\chatgpt[]",
    "llama":   r"\llama[]",
    "mistral": r"\mistral[]",
    "gptoss":  r"\ossgpt[]",
    "ruleid":  r"\ruleid[]",
}

def parse_pct(s):
    return float(s.strip().rstrip("%"))

def parse_val(s):
    return float(s.strip())

def load_data(path):
    data = {}  # (classifier, semantic_model) -> {auprc, auroc, auprc_ci, auroc_ci}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["model"].strip()
            if key not in MODEL_MAP:
                continue
            clf, sem = MODEL_MAP[key]
            data[(clf, sem)] = {
                "auprc":    parse_val(row["auprc"])   * 100,
                "auroc":    parse_val(row["rocauc"])  * 100,
                "auprc_ci": parse_val(row["auprc_ci"]) * 100,
                "auroc_ci": parse_val(row["auroc_ci"]) * 100,
            }
    return data

def fmt_ci(v):
    # Round to 2 decimal places; show as e.g. 0.03
    return f"{v:.2f}"

def make_cell(value, ci, bold):
    ci_str  = fmt_ci(ci)
    val_str = f"{value:.2f}$\\pm${ci_str}"
    return f"\\textbf{{{val_str}}}" if bold else val_str

def generate_table(data):
    lines = []

    lines += [
        r"\begin{table}[htb]",
        r"  \centering",
        r"  \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{c|c|ccccccc}\hline",
    ]

    # Header
    col_headers = " &\n      ".join(f"\\textbf{{{LATEX_LABELS[m]}}}" for m in SEMANTIC_MODELS)
    lines += [
        r"      \textbf{Classifier} & \textbf{Metric} &",
        f"      {col_headers} \\\\ \\hline",
    ]

    for clf in CLASSIFIERS:
        for metric in ["auprc", "auroc"]:
            metric_key = "auprc" if metric == "auprc" else "auroc"
            ci_key     = "auprc_ci" if metric == "auprc" else "auroc_ci"

            values = {sem: data[(clf, sem)][metric_key] for sem in SEMANTIC_MODELS}
            cis    = {sem: data[(clf, sem)][ci_key]     for sem in SEMANTIC_MODELS}
            max_v  = max(values.values())

            cells = " & ".join(
                make_cell(values[sem], cis[sem], bold=(values[sem] == max_v))
                for sem in SEMANTIC_MODELS
            )
            metric_label = "AUPRC" if metric == "auprc" else "AUROC"

            if metric == "auprc":
                lines.append(
                    f"      \\multirow{{2}}{{*}}{{\\shortstack{{{clf}}}}} & {{{metric_label}}} & {cells} \\\\"
                )
            else:
                suffix = r" \\ \hline" if clf == "OCSVM" else r" \\"
                lines.append(f"      & {{{metric_label}}} & {cells}{suffix}")

    lines += [
        r"    \end{tabular}",
        r"  }",
        r"  \caption{AUPRC and AUROC (\%) for SQL attack detection with 95\% confidence intervals of \gaur{} using different semantic models}",
        r"  \label{tab:performance-nov}",
        r"\end{table}",
    ]
    return "\n".join(lines)

if __name__ == "__main__":
    data = load_data(CSV_PATH)
    print(generate_table(data))
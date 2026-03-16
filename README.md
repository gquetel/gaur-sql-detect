# GAUR-SQL-Detect

Code for the evaluations presented in our paper: **Parser Instrumentation for Semantic-Aware Applicative Intrusion Detection**. Published in 41st International Conference on ICT Systems Security and Privacy Protection (IFIPSEC26), 2026. Recommended citation:

> Quetel, G., Gimenez, P. F., Robert, T. & Pautet, L. (2026 June). Parser Instrumentation for Semantic-Aware Applicative Intrusion Detection. In the 41st International Conference on ICT Systems Security and Privacy Protection (IFIPSEC26) 

This repository relies on instrumented MySQL servers built with [gaur](https://github.com/gquetel/gaur), available from [gaur-instrumented-apps](https://github.com/gquetel/gaur-instrumented-apps).

---

## Environment

Our experiments heavily relies on the [Nix](https://nixos.org/download) build system. [gaur-instrumented-apps](https://github.com/gquetel/gaur-instrumented-apps) provide derivations to build, initialize and run instrumented MySQLs. This codebase requires access to instrumented MySQLs. Use [shell.nix](./shell.nix) to enter a reproducible and pinned development environment. 

Alternatively, you could manually patch, build, initialize and run a MySQL server then install the required dependencies using:  

```bash
pip install -e .
```

Then, to be able to save figures using plotly and kaleido. Run: 
```bash
plotly_get_chrome
```

### Dataset
We used the [Superviz25-SQL](https://zenodo.org/records/17086037) dataset. By default, it is expected at `./data/dataset.csv`.

---

## Usage 

###  Obtaining features from GAUR 

We tried to provide a self-sufficient public API for obtaining features from `gaur` instrumented MySQL. The following code snippet allows you to obtain such features from a given DataFrame (it must contain a `full_query` column). 

```python
import gaur_sqld
import pandas as pd

# Optional: override server prefix (default: ~/tmp/)
gaur_sqld.configure(prefix="~/tmp/")

df = pd.read_csv("data/dataset.csv")
traces = gaur_sqld.get_traces_from_df(df)
features = gaur_sqld.pre_process_for_gaur(traces, mode="expert")
```

`get_traces_from_df()` first checks whether a socket already exists at the expected path (`<prefix>/<hostname>/mysqld-<trace_type>/socket`). If it does, the running server is used. If no socket is found and Nix is available, the library will automatically:

1. Clone `gaur-instrumented-apps` to `~/.local/share/gaur-sqld/gaur-instrumented-apps/`
2. Build the instrumented MySQL server with `nix-build`
3. Start the server and wait for its socket

If neither condition is met, a `GaurServerError` is raised with instructions for manual setup.

---

### Configuration

By default the bundled `config/config.toml` is used. Pass a custom TOML file with the `--config` flag:

```bash
gaur-sql-detect --config /path/to/my.toml
```

The `prefix` key in `[mysql]` sets the directory that contains the instrumented server instances (sockets and data directories). Default: `~/tmp/`.

Programmatic override:

```python
gaur_sqld.configure(prefix="/my/servers/", trace_type="expert", seed=2)
# or load a full config file:
gaur_sqld.configure_from_file("/path/to/my.toml")
```

---


## Experiments

To reproduce the result of the paper, you can use the script :

```bash
python3 run_eval.py --models ae --trace-type all
python3 run_eval.py --models ocsvm --trace-type all
```

Supplementary experiments are located in [experiments/](experiments/).

### llm-tagging

Material for the LLM-based semantic tagging of MySQL grammar rules: rule list, parse-tree generation script, and the prompts used.

### overhead-dc

Measure the data-collection overhead of GAUR:

```bash
cd experiments/overhead-dc/
nix-build default.nix
./result/bin/overhead-experiment
```

Without Nix, run `overhead.py` directly (see `--help` for arguments).

### overhead-inference

Notebook generating the AUROC vs. average inference time figure. Requires `results.csv` and `inference.csv` produced by the main evaluation script.

### semantic_mutation

Robustness evaluation using [WAF-A-MoLE](https://github.com/AvalZ/WAF-A-MoLE). Requires trained model weights and the Superviz25-SQL dataset.

---
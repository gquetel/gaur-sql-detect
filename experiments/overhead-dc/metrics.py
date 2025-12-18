import pandas as pd
import numpy as np
from scipy import stats

# Cols: query,timen,timec,overhead
# timen = normal mysql (baseline)
# timec = custom mysql (instrumented)

FILENAME = "/home/gquetel/experiences-results/2025-11-24-overhead-runtime-anubis/overhead_stats_final.csv"

# Load the CSV
df = pd.read_csv(FILENAME)

cutoff = df["overhead"].quantile(0.999)
print(f"0.1% highest overhead starts at {cutoff * 1000:.6f} ms")
print(len(df)*0.001)


cutoff = df["overhead"].quantile(0.001)
print(f"0.1% lowest overhead starts at {cutoff * 1000:.6f} ms")

# Filter queries with overhead > cutoff
df_high = df[df["overhead"] > cutoff]
# df_high.to_csv("cutoff.csv")
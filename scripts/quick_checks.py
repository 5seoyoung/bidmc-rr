# scripts/quick_checks.py
import sys, numpy as np, pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/windows.parquet"
df = pd.read_parquet(path)

print("shape:", df.shape); print(df.head(2), "\n")
print("subjects:", df["subject"].nunique())
print("rows:", len(df), "\n")

print(df[["sqi","rr","t_start","t_end"]].describe(), "\n")

bad_rr = df[(df["rr"].isna()) | (df["rr"] < 4) | (df["rr"] > 40)]
print("bad_rr rows:", len(bad_rr), "\n")

print("sqi quantiles:\n", df["sqi"].quantile([0, .25, .5, .75, .9, .95, 1]), "\n")

by_subj = df.groupby("subject").agg(
    n=("rr","size"),
    n_nan_rr=("rr", lambda s: s.isna().sum()),
    rr_median=("rr","median"),
    sqi_median=("sqi","median"),
)
print(by_subj.sort_index().to_string())
print("\nFlagged(51–53):\n", by_subj.loc[by_subj.index.intersection([51,52,53])])

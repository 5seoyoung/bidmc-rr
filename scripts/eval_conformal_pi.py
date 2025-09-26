#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.path)
    need = {"rr","rr_hat_freq","pi_lo","pi_hi","subject"}
    if not need.issubset(df.columns):
        raise ValueError(f"{args.path} must have {need}")

    inside = (df["rr"] >= df["pi_lo"]) & (df["rr"] <= df["pi_hi"])
    coverage = float(np.nanmean(inside))
    width = float(np.nanmean(df["pi_hi"] - df["pi_lo"]))
    mae = float(np.nanmean(np.abs(df["rr_hat_freq"] - df["rr"])))

    print(f"=== {args.label.upper()} Conformal ===")
    print({"coverage": round(coverage,3), "mean_width": round(width,3), "MAE_point": round(mae,3)})

    # subject별 커버리지/폭
    sub = df.groupby("subject").apply(
        lambda g: pd.Series({
            "n": len(g),
            "coverage": np.nanmean((g["rr"] >= g["pi_lo"]) & (g["rr"] <= g["pi_hi"])),
            "mean_width": np.nanmean(g["pi_hi"] - g["pi_lo"]),
            "MAE_point": np.nanmean(np.abs(g["rr_hat_freq"] - g["rr"])),
        })
    ).reset_index()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    sub.to_csv(Path(args.outdir, f"{args.label}_conformal_by_subject.csv"), index=False)
    print(f"Saved: {Path(args.outdir, f'{args.label}_conformal_by_subject.csv')}")

if __name__ == "__main__":
    main()

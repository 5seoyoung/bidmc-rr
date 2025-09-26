#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def robust_z(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)

def bin_edges_from_cal(x_cal, n_bins):
    qs = np.linspace(0,1,n_bins+1)
    edges = np.nanquantile(x_cal, qs)
    edges[0] = -np.inf; edges[-1] = np.inf
    return edges

def digitize_with_edges(x, edges):
    # returns bin index in [0, len(edges)-2]
    idx = np.digitize(x, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges)-2)
    return idx

def split_conformal_quantile(abs_err, alpha):
    n = len(abs_err)
    if n == 0: return np.nan
    r = int(np.ceil((n + 1) * (1 - alpha))) - 1
    r = np.clip(r, 0, n-1)
    arr = np.sort(abs_err)
    return float(arr[r])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)  # 90%
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--feature", choices=["zsnr","sqi"], default="zsnr")
    ap.add_argument("--n-bins", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.path).copy()
    need = {"subject","rr","rr_hat_freq"}
    if not need.issubset(df.columns):
        raise ValueError(f"{args.path} must have {need}")

    if args.feature == "zsnr":
        if "psd_snr_db" not in df.columns:
            raise ValueError("psd_snr_db missing.")
        df["feature_val"] = robust_z(df["psd_snr_db"])
    else:
        if "sqi" not in df.columns:
            raise ValueError("sqi missing.")
        df["feature_val"] = df["sqi"].astype(float)

    df["abs_err"] = np.abs(df["rr_hat_freq"] - df["rr"])
    gkf = GroupKFold(n_splits=args.n_folds)
    groups = df["subject"].to_numpy()

    q_assign = np.full(len(df), np.nan)

    for tr, te in gkf.split(df, groups=groups, y=None):
        cal = df.iloc[tr].copy()
        te_df = df.iloc[te].copy()

        edges = bin_edges_from_cal(cal["feature_val"].to_numpy(), args.n_bins)
        cal["bin"] = digitize_with_edges(cal["feature_val"].to_numpy(), edges)
        te_bins = digitize_with_edges(te_df["feature_val"].to_numpy(), edges)

        # bin별 q
        qs = {}
        for b in range(args.n_bins):
            q = split_conformal_quantile(
                cal.loc[cal["bin"]==b, "abs_err"].dropna().to_numpy(),
                args.alpha
            )
            if np.isnan(q):
                # fallback: 전체
                q = split_conformal_quantile(cal["abs_err"].dropna().to_numpy(), args.alpha)
            qs[b] = q

        q_assign[te] = [qs[int(b)] for b in te_bins]

    df["pi_q"] = q_assign
    df["pi_lo"] = df["rr_hat_freq"] - df["pi_q"]
    df["pi_hi"] = df["rr_hat_freq"] + df["pi_q"]

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()

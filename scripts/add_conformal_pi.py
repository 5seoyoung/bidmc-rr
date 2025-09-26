#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)  # 90% PI
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.path).copy()
    need = {"subject","rr","rr_hat_freq"}
    if not need.issubset(df.columns):
        raise ValueError(f"{args.path} must have {need}")
    df["abs_err"] = np.abs(df["rr_hat_freq"] - df["rr"])

    # 그룹 KFold로 subject-wise conformal (single quantile)
    gkf = GroupKFold(n_splits=args.n_folds)
    groups = df["subject"].to_numpy()
    qs = np.full(len(df), np.nan)

    for tr, te in gkf.split(df, groups=groups, y=None):
        # 학습 없이, 보정만: 칼리브 세트 = tr
        cal_err = df.iloc[tr]["abs_err"].dropna().to_numpy()
        if len(cal_err) == 0:
            q = np.nan
        else:
            # 이론상 q = (1-alpha)*(1+1/n) 분위 (split conformal 보정)
            n = len(cal_err)
            rank = int(np.ceil((n + 1) * (1 - args.alpha))) - 1
            rank = np.clip(rank, 0, n - 1)
            q = np.partition(np.sort(cal_err), rank)[rank]
        qs[te] = q

    df["pi_q"] = qs
    df["pi_lo"] = df["rr_hat_freq"] - df["pi_q"]
    df["pi_hi"] = df["rr_hat_freq"] + df["pi_q"]

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()

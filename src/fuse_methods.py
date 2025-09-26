#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def robust_z(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)

def softmax(a, tau=1.0):
    a = np.asarray(a, float)
    a = a / max(1e-12, float(tau))
    a = a - np.nanmax(a)  # 안정화
    w = np.exp(a)
    w[np.isnan(w)] = 0.0
    s = w.sum()
    return w / s if s > 0 else np.zeros_like(w)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--rule", choices=["argmax", "softmax"], default="softmax")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.paths) == len(args.labels)

    # 키로 머지
    base_cols = ["subject","t_start","t_end","sqi","rr"]
    dfs = []
    for p, lab in zip(args.paths, args.labels):
        df = pd.read_parquet(p)
        miss = set(base_cols) - set(df.columns)
        if miss:
            raise ValueError(f"{p} missing {miss}")
        need_pred = {"rr_hat_freq", "psd_snr_db"}
        if not need_pred.issubset(df.columns):
            raise ValueError(f"{p} must have {need_pred}")
        df = df[base_cols + ["rr_hat_freq","psd_snr_db"]].copy()
        df = df.rename(columns={
            "rr_hat_freq": f"rr_{lab}",
            "psd_snr_db": f"snr_{lab}",
        })
        dfs.append(df)

    # 순차 머지
    out = dfs[0]
    for df in dfs[1:]:
        out = out.merge(df, on=base_cols, how="inner")

    # 각 방법 snr에 robust z-score
    zcols = []
    for lab in args.labels:
        z = robust_z(out[f"snr_{lab}"])
        out[f"zsnr_{lab}"] = z
        zcols.append(f"zsnr_{lab}")

    # 융합
    rr_fused = np.full(len(out), np.nan)
    snr_fused = np.full(len(out), np.nan)  # 선택/가중 반영
    if args.rule == "argmax":
        zmat = out[zcols].to_numpy()
        idx = np.nanargmax(zmat, axis=1)
        for i, j in enumerate(idx):
            lab = args.labels[j]
            rr_fused[i] = out.at[i, f"rr_{lab}"]
            snr_fused[i] = out.at[i, f"snr_{lab}"]
    else:
        # softmax(정규화 SNR)로 가중 평균
        zmat = out[zcols].to_numpy()
        rr_mat = out[[f"rr_{lab}" for lab in args.labels]].to_numpy()
        snr_mat = out[[f"snr_{lab}" for lab in args.labels]].to_numpy()
        for i in range(len(out)):
            zs = zmat[i]
            rrs = rr_mat[i]
            snrs = snr_mat[i]
            # 유효한 것만
            ok = np.isfinite(zs) & np.isfinite(rrs)
            if not np.any(ok): continue
            w = softmax(zs[ok], tau=args.temperature)
            rr_fused[i] = float(np.sum(w * rrs[ok]))
            snr_fused[i] = float(np.sum(w * snrs[ok]))

    out["rr_hat_freq"] = rr_fused
    out["psd_snr_db"] = snr_fused
    # 최종 컬럼 정리
    keep = base_cols + ["rr_hat_freq","psd_snr_db"]
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    out[keep].to_parquet(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()

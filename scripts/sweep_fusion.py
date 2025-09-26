#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, itertools
from pathlib import Path
import numpy as np
import pandas as pd

BASE_COLS = ["subject","t_start","t_end","sqi","rr"]

def robust_z(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)

def softmax(a, tau=1.0):
    a = np.asarray(a, float)
    a = a / max(1e-12, float(tau))
    a = a - np.nanmax(a)
    w = np.exp(a)
    w[np.isnan(w)] = 0.0
    s = w.sum()
    return w / s if s > 0 else np.zeros_like(w)

def load_and_merge(paths, labels):
    dfs = []
    for p, lab in zip(paths, labels):
        df = pd.read_parquet(p)
        need_pred = {"rr_hat_freq","psd_snr_db"}
        miss = set(BASE_COLS) - set(df.columns)
        if miss: raise ValueError(f"{p} missing {miss}")
        if not need_pred.issubset(df.columns):
            raise ValueError(f"{p} must have {need_pred}")
        df = df[BASE_COLS + ["rr_hat_freq","psd_snr_db"]].copy()
        df = df.rename(columns={"rr_hat_freq": f"rr_{lab}", "psd_snr_db": f"snr_{lab}"})
        dfs.append(df)
    out = dfs[0]
    for df in dfs[1:]:
        out = out.merge(df, on=BASE_COLS, how="inner")
    return out

def fuse(df, labels, rule="softmax", tau=0.8, max_w=None, cap_label=None):
    # 정규화 SNR
    zcols = []
    for lab in labels:
        df[f"zsnr_{lab}"] = robust_z(df[f"snr_{lab}"])
        zcols.append(f"zsnr_{lab}")
    zmat = df[zcols].to_numpy()
    rr_mat = df[[f"rr_{lab}" for lab in labels]].to_numpy()
    snr_mat = df[[f"snr_{lab}" for lab in labels]].to_numpy()

    rr_fused = np.full(len(df), np.nan)
    snr_fused = np.full(len(df), np.nan)

    for i in range(len(df)):
        zs, rrs, snrs = zmat[i], rr_mat[i], snr_mat[i]
        ok = np.isfinite(zs) & np.isfinite(rrs)
        if not np.any(ok): continue
        if rule == "argmax":
            j = np.nanargmax(zs[ok])
            jg = np.where(ok)[0][j]
            rr_fused[i] = rrs[jg]; snr_fused[i] = snrs[jg]
        else:
            w = softmax(zs[ok], tau=tau)
            # 옵션: 특정 방법 가중 상한 (예: PRV가 먹어치우는 것 방지)
            if max_w is not None and cap_label is not None and cap_label in labels:
                idx_cap = np.where(np.array(labels)[ok] == cap_label)[0]
                if len(idx_cap) == 1:
                    jcap = idx_cap[0]
                    if w[jcap] > max_w:
                        # cap 적용: cap 대상은 max_w로 고정, 나머지 합을 (1 - max_w)로 스케일
                        orig = w[jcap]
                        w[jcap] = max_w
                        other_idx = [k for k in range(len(w)) if k != jcap]
                        other_sum = w[other_idx].sum()
                        if other_sum > 0:
                            scale = (1.0 - max_w) / other_sum
                            w[other_idx] *= scale
                        else:
                            # cap 대상만 살아있다면 cap을 1로 승격(사실상 argmax)
                            w[:] = 0.0
                            w[jcap] = 1.0
            rr_fused[i] = float(np.sum(w * rrs[ok]))
            snr_fused[i] = float(np.sum(w * snrs[ok]))
    fused = df[BASE_COLS].copy()
    fused["rr_hat_freq"] = rr_fused
    fused["psd_snr_db"] = snr_fused
    return fused

def metrics(df):
    e = np.abs(df["rr_hat_freq"] - df["rr"])
    return {
        "n": int(e.notna().sum()),
        "MAE": float(np.nanmean(e)),
        "MdAE": float(np.nanmedian(e)),
        "RMSE": float(np.sqrt(np.nanmean((df["rr_hat_freq"]-df["rr"])**2)))
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--rules", nargs="+", default=["argmax","softmax"])
    ap.add_argument("--taus", type=str, default="0.2,0.4,0.6,0.8,1.0,2.0")
    ap.add_argument("--cap-label", type=str, default=None, help="가중 상한을 적용할 방법 라벨(예: prv)")
    ap.add_argument("--max-w", type=float, default=None, help="cap-label의 최대 가중(예: 0.5)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--best-out", required=True)
    args = ap.parse_args()
    assert len(args.paths)==len(args.labels)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    dfm = load_and_merge(args.paths, args.labels)
    taus = [float(t) for t in args.taus.split(",") if t.strip()]

    rows = []
    best = None
    for rule in args.rules:
        if rule == "argmax":
            fused = fuse(dfm, args.labels, rule="argmax")
            m = metrics(fused)
            m.update({"rule": rule, "tau": None, "cap_label": None, "max_w": None})
            rows.append(m)
            if best is None or m["MAE"] < best["cfg"]["MAE"]:
                best = {"cfg": m, "df": fused}
        else:
            for tau in taus:
                fused = fuse(dfm, args.labels, rule="softmax", tau=tau,
                            max_w=args.max_w, cap_label=args.cap_label)
                m = metrics(fused)
                m.update({"rule": rule, "tau": tau, "cap_label": args.cap_label, "max_w": args.max_w})
                rows.append(m)
                if best is None or m["MAE"] < best["cfg"]["MAE"]:
                    best = {"cfg": m, "df": fused}

    summ = pd.DataFrame(rows).sort_values("MAE")
    summ.to_csv(Path(args.outdir, "fusion_sweep_summary.csv"), index=False)
    print(summ.head(10).to_string(index=False))

    best_df = best["df"]
    best_df.to_parquet(args.best_out, index=False)
    print(f"Best fusion saved: {args.best_out}")
    print("Best config:", best["cfg"])

if __name__ == "__main__":
    main()

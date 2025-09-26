#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

def metrics(df):
    err = np.abs(df["rr_hat_freq"] - df["rr"])
    return {
        "n": int(err.notna().sum()),
        "MAE": float(np.nanmean(err)),
        "MdAE": float(np.nanmedian(err)),
        "RMSE": float(np.sqrt(np.nanmean((df["rr_hat_freq"] - df["rr"])**2))),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="parquet with rr_hat_freq, rr, sqi, psd_snr_db")
    ap.add_argument("--label", default="model")
    ap.add_argument("--min-coverage", type=float, default=0.60)
    ap.add_argument("--outdir", default="data/derived")
    ap.add_argument("--sqi-dir", choices=["high", "low"], default="high",
                    help="SQI가 높을수록 좋은 경우 'high'(sqi>=thr), 반대면 'low'(sqi<=thr)")
    ap.add_argument("--snr-range", type=float, nargs=3, default=[-2.0, 12.0, 0.5],
                    help="SNR dB threshold sweep: start stop step")
    args = ap.parse_args()

    df = pd.read_parquet(args.path)
    for c in ("rr_hat_freq","rr","sqi","psd_snr_db"):
        if c not in df.columns:
            raise ValueError(f"{args.path} missing column: {c}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base = metrics(df)
    N = len(df)
    base["coverage"] = 1.0
    print(f"\n=== {args.label.upper()} (no gating) ===")
    print({k: (round(v,3) if isinstance(v,float) else v) for k,v in base.items()})

    # SQI thresholds as quantiles
    qs = np.linspace(0.0, 1.0, 21)
    thr_sqi_vals = np.quantile(df["sqi"].to_numpy(), qs)
    snr_start, snr_stop, snr_step = args.snr_range
    thr_snr_vals = np.arange(snr_start, snr_stop + 1e-9, snr_step)

    grid_rows = []
    best = None

    for thr_sqi in thr_sqi_vals:
        if args.sqi_dir == "high":
            m_sqi = df["sqi"] >= thr_sqi
            mode_tag = "sqi>=thr"
        else:
            m_sqi = df["sqi"] <= thr_sqi
            mode_tag = "sqi<=thr"

        for thr_snr in thr_snr_vals:
            m_snr = df["psd_snr_db"] >= thr_snr
            m = m_sqi & m_snr & df["rr_hat_freq"].notna() & df["rr"].notna()
            kept = df[m]
            cov = len(kept) / N
            if cov <= 0:
                row = {"coverage": 0.0, "MAE": np.inf, "MdAE": np.inf, "RMSE": np.inf,
                       "thr_sqi": float(thr_sqi), "thr_snr": float(thr_snr)}
                grid_rows.append(row); continue
            mm = metrics(kept)
            row = {"coverage": cov, "thr_sqi": float(thr_sqi), "thr_snr": float(thr_snr), **mm}
            grid_rows.append(row)

            if cov >= args.min_coverage:
                if (best is None) or (mm["MAE"] < best["MAE"]):
                    best = {"mode": f"{mode_tag} & snr>=thr",
                            "coverage": cov, "thr_sqi": float(thr_sqi),
                            "thr_snr": float(thr_snr), **mm}

    grid = pd.DataFrame(grid_rows)
    grid_path = outdir / f"{args.label}_sqi_snr_grid.csv"
    grid.to_csv(grid_path, index=False)

    if best is not None:
        best_fmt = {k: (round(v,3) if isinstance(v,float) else v) for k,v in best.items()}
        print(f"\n>>> Best 2D gate (coverage >= {args.min_coverage}):")
        print(best_fmt)
    else:
        print(f"\n>>> No combination met coverage >= {args.min_coverage}")

    # Heatmap (MAE) at each coverage bucket (e.g., >= min_coverage)
    try:
        g2 = grid[grid["coverage"] >= args.min_coverage].copy()
        if len(g2):
            piv = g2.pivot_table(index="thr_snr", columns="thr_sqi", values="MAE")
            fig = plt.figure(figsize=(8,5))
            im = plt.imshow(piv.values, aspect="auto", origin="lower")
            plt.colorbar(im, label="MAE")
            plt.yticks(np.arange(len(piv.index)), [f"{v:.1f}" for v in piv.index])
            plt.xticks(np.arange(len(piv.columns)), [f"{v:.3f}" for v in piv.columns], rotation=90)
            plt.title(f"{args.label.upper()} — MAE heatmap (coverage ≥ {args.min_coverage})")
            plt.xlabel("thr_sqi"); plt.ylabel("thr_snr(dB)")
            fig.tight_layout()
            fig_path = outdir / f"{args.label}_sqi_snr_heatmap.png"
            fig.savefig(fig_path, dpi=160)
            plt.close(fig)
    except Exception as e:
        print(f"[warn] heatmap skipped: {e}")

    print(f"\nSaved: {grid_path}")

if __name__ == "__main__":
    main()

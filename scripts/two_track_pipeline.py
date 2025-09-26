#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-track evaluation pipeline for RR with conformal intervals.

Tracks
------
A) All-comers:
    - input: fused_best parquet (no gating)
    - CP: Mondrian by zSNR (3 bins), alpha=0.08, 5-fold GroupKFold(by subject)
    - outputs: parquet(with pi_lo/pi_hi), metrics csv, width-gating sweep/plots

B) High-confidence:
    - 2D gate: (SQI >= 0.048) & (SNR >= -2.0)
    - re-calibrate CP on gated subset only (Mondrian, alpha=0.10)
    - outputs: gated parquet(with pi_lo/pi_hi), metrics csv, width-gating sweep/plots

Notes
-----
- Auto-detects prediction column: prefer 'rr_hat_freq', else 'rr_hat', else first 'rr_hat*'.
- zSNR = robust z-score of psd_snr_db (median/MAD). Falls back to standard z if MAD≈0.
- Uses GroupKFold by 'subject' when available, otherwise KFold.
- Mondrian bins = quantile bins on the calibration fold (consistent across its test fold).

Usage
-----
python scripts/two_track_pipeline.py \
  --path data/processed/windows_with_pred_fused_best.parquet \
  --outdir data/derived \
  --alpha-all 0.08 --alpha-gated 0.10 \
  --thr-sqi 0.048 --thr-snr -2.0 \
  --n-bins 3 --n-folds 5

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List

# --------------- small utils ---------------

def detect_pred_col(df: pd.DataFrame) -> str:
    if "rr_hat_freq" in df.columns:
        return "rr_hat_freq"
    if "rr_hat" in df.columns:
        return "rr_hat"
    # fallback to any rr_hat* column
    cands = [c for c in df.columns if c.startswith("rr_hat")]
    assert len(cands) > 0, f"No rr_hat* column in DataFrame. columns={list(df.columns)[:20]}"
    return cands[0]

def ensure_cols(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad <= 1e-9 or not np.isfinite(mad):
        mu = np.nanmean(x)
        sd = np.nanstd(x) + 1e-9
        return (x - mu) / sd
    return (x - med) / (1.4826 * mad + 1e-9)

def split_quantile(residuals: np.ndarray, alpha: float) -> float:
    """
    Split conformal quantile: ceil((n+1)*(1-alpha)) / n
    Implemented by sorting and taking kth order statistic.
    """
    r = np.asarray(residuals)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return np.nan
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = max(1, min(k, n))
    r_sorted = np.sort(r)
    return float(r_sorted[k - 1])

def mondrian_bins_fit(x_cal: np.ndarray, n_bins: int):
    """Return bin edges (quantile-based) computed on calibration feature."""
    # Use quantile edges: n_bins => edges of length n_bins+1
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x_cal, qs))
    # ensure strictly increasing (de-dup) – if too collapsed, fallback to min/max
    if len(edges) < 2:
        lo, hi = float(np.nanmin(x_cal)), float(np.nanmax(x_cal))
        edges = np.array([lo, hi], dtype=float)
    # pad a hair to include rightmost
    edges[0] = np.floor(edges[0] * 1000.0) / 1000.0 - 1e-6
    edges[-1] = np.ceil(edges[-1] * 1000.0) / 1000.0 + 1e-6
    return edges

def mondrian_bins_apply(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return bin index per sample using prefit edges (0..B-1)."""
    # np.digitize: returns 1..len(edges)-1; subtract 1 to get 0-based
    return np.clip(np.digitize(x, edges, right=True) - 1, 0, len(edges) - 2)

def group_kfold_indices(n_splits: int, groups: Optional[np.ndarray], n_samples: int, seed: int = 42):
    """
    Yield (train_idx, cal_idx) for split-conformal (use cal_idx for calibration, train_idx~test_idx)
    Here we don't train a model; we only need to split data into 'cal' and 'test' folds.
    We'll just treat 'train_idx' as 'test_idx' to fill predictions; the predictor is fixed beforehand.
    """
    if groups is not None:
        try:
            from sklearn.model_selection import GroupKFold
            gkf = GroupKFold(n_splits=n_splits)
            for cal_idx, test_idx in gkf.split(np.arange(n_samples), groups=groups):
                yield (np.setdiff1d(np.arange(n_samples), cal_idx), cal_idx)  # (test, cal) order not vital here
            return
        except Exception:
            pass
    # fallback KFold (shuffle for robustness)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for cal_idx, test_idx in kf.split(np.arange(n_samples)):
        yield (test_idx, cal_idx)

def add_conformal_pi_mondrian(
    df: pd.DataFrame,
    y_col: str,
    yhat_col: str,
    feature_col: str,
    n_bins: int,
    alpha: float,
    n_folds: int = 5,
    group_col: Optional[str] = "subject",
) -> pd.DataFrame:
    """
    Split-conformal Mondrian intervals by feature_col.
    Returns a copy of df with 'pi_lo','pi_hi' columns filled.
    """
    assert y_col in df and yhat_col in df and feature_col in df, "Required columns missing."
    out = df.copy()
    out["pi_lo"] = np.nan
    out["pi_hi"] = np.nan

    y = out[y_col].to_numpy()
    yhat = out[yhat_col].to_numpy()
    feat = out[feature_col].to_numpy()
    groups = out[group_col].to_numpy() if (group_col in out.columns) else None
    n = len(out)

    for test_idx, cal_idx in group_kfold_indices(n_folds, groups, n):
        # calibration sets binning on feature
        feat_cal = feat[cal_idx]
        edges = mondrian_bins_fit(feat_cal, n_bins=n_bins)
        cal_bins = mondrian_bins_apply(feat_cal, edges)

        # per-bin residual quantiles
        res = np.abs(y[cal_idx] - yhat[cal_idx])
        bin_q = {}
        for b in np.unique(cal_bins):
            rb = res[cal_bins == b]
            q = split_quantile(rb, alpha=alpha)
            if not np.isfinite(q):
                q = split_quantile(res, alpha=alpha)
            bin_q[int(b)] = float(q)

        # apply to test fold (we just fill intervals there)
        test_bins = mondrian_bins_apply(feat[test_idx], edges)
        q_per = np.array([bin_q.get(int(b), split_quantile(res, alpha=alpha)) for b in test_bins], dtype=float)
        lo = yhat[test_idx] - q_per
        hi = yhat[test_idx] + q_per
        out.loc[out.index[test_idx], "pi_lo"] = lo
        out.loc[out.index[test_idx], "pi_hi"] = hi

    return out

def basic_metrics(df: pd.DataFrame, y_col: str, yhat_col: str) -> Dict[str, float]:
    n = int(len(df))
    mae = float(np.nanmean(np.abs(df[yhat_col] - df[y_col])))
    md = float(np.nanmedian(np.abs(df[yhat_col] - df[y_col])))
    rmse = float(np.sqrt(np.nanmean((df[yhat_col] - df[y_col])**2)))
    return {"n": n, "MAE": round(mae, 3), "MdAE": round(md, 3), "RMSE": round(rmse, 3)}

def pi_metrics(df: pd.DataFrame, y_col: str) -> Dict[str, float]:
    keep = df.dropna(subset=["pi_lo", "pi_hi", y_col])
    n = int(len(keep))
    cov = float(np.mean((keep[y_col] >= keep["pi_lo"]) & (keep[y_col] <= keep["pi_hi"])))
    width = float(np.mean(keep["pi_hi"] - keep["pi_lo"]))
    return {"coverage": round(cov, 3), "mean_width": round(width, 3), "n": n}

def width_tradeoff(
    df: pd.DataFrame,
    y_col: str,
    yhat_col: str,
    label: str,
    outdir: str,
    w_min: float = 5.0,
    w_max: float = 80.0,
    n_steps: int = 16,
):
    dd = df.dropna(subset=["pi_lo", "pi_hi", y_col]).copy()
    dd["width"] = dd["pi_hi"] - dd["pi_lo"]
    rows = []
    for w in np.linspace(w_min, w_max, n_steps):
        keep = dd[dd["width"] <= w]
        if len(keep) == 0:
            continue
        cov = np.mean((keep[y_col] >= keep["pi_lo"]) & (keep[y_col] <= keep["pi_hi"]))
        mae = np.mean(np.abs(keep[yhat_col] - keep[y_col]))
        rows.append({
            "w_thr": float(w),
            "n": int(len(keep)),
            "retention": float(len(keep) / len(dd)),
            "coverage": float(cov),
            "MAE": float(mae),
        })
    out = pd.DataFrame(rows).sort_values("w_thr")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{label}_width_tradeoff.csv")
    out.to_csv(csv_path, index=False)

    # plots
    plt.figure()
    plt.plot(out["retention"], out["coverage"], marker="o")
    plt.xlabel("Retention"); plt.ylabel("Coverage")
    plt.title(f"{label}: Coverage vs Retention (width gating)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_width_cov_ret.png"), dpi=200)

    plt.figure()
    plt.plot(out["retention"], out["MAE"], marker="o")
    plt.xlabel("Retention"); plt.ylabel("MAE")
    plt.title(f"{label}: MAE vs Retention (width gating)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{label}_width_mae_ret.png"), dpi=200)

    # report minimum width achieving >=0.90 coverage, if any
    ok = out[out["coverage"] >= 0.90]
    if len(ok):
        w_star = ok.sort_values("w_thr").iloc[0].to_dict()
        print(f">>> {label} min width for coverage >= 0.90:", {k: (float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in w_star.items()})
    else:
        print(f">>> {label} no width in sweep achieved coverage >= 0.90")

# --------------- main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to fused_best parquet (predictions & SQI/SNR present).")
    ap.add_argument("--outdir", default="data/derived")
    ap.add_argument("--alpha-all", type=float, default=0.08, help="alpha for All-comers CP")
    ap.add_argument("--alpha-gated", type=float, default=0.10, help="alpha for High-confidence CP")
    ap.add_argument("--n-bins", type=int, default=3, help="Mondrian bins")
    ap.add_argument("--n-folds", type=int, default=5, help="GroupKFold splits")
    ap.add_argument("--thr-sqi", type=float, default=0.048)
    ap.add_argument("--thr-snr", type=float, default=-2.0)
    ap.add_argument("--label", default="fused_best", help="base label for outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_parquet(args.path)
    # required columns
    pred_col = detect_pred_col(df)
    ensure_cols(df, ["rr", "sqi", "psd_snr_db"])
    if "subject" not in df.columns:
        # create a dummy subject if not present
        df["subject"] = 0

    # add zsnr if missing
    if "zsnr" not in df.columns:
        df["zsnr"] = robust_z(df["psd_snr_db"].to_numpy())

    # ---------- Track A: All-comers ----------
    label_all = f"{args.label}_all_a{str(args.alpha_all).replace('.','')}"
    print(f"\n[Track A] All-comers :: alpha={args.alpha_all}, bins={args.n_bins}, folds={args.n_folds}")
    dfa = df.copy()
    dfa = dfa.dropna(subset=["rr", pred_col, "zsnr"]).copy()
    dfa_cp = add_conformal_pi_mondrian(
        dfa, y_col="rr", yhat_col=pred_col,
        feature_col="zsnr", n_bins=args.n_bins, alpha=args.alpha_all,
        n_folds=args.n_folds, group_col="subject"
    )
    # save parquet
    out_parquet_a = os.path.join("data/processed", f"windows_with_pred_{label_all}.parquet")
    os.makedirs(os.path.dirname(out_parquet_a), exist_ok=True)
    dfa_cp.to_parquet(out_parquet_a, index=False)
    # metrics
    base_metrics_a = basic_metrics(dfa_cp, "rr", pred_col)
    pi_m_a = pi_metrics(dfa_cp, "rr")
    pd.DataFrame([{
        **{"label": label_all, "mode": "all-comers", "pred_col": pred_col},
        **base_metrics_a, **pi_m_a
    }]).to_csv(os.path.join(args.outdir, f"{label_all}_metrics.csv"), index=False)
    print("All-comers (point):", base_metrics_a)
    print("All-comers (PI):   ", pi_m_a)
    width_tradeoff(dfa_cp, "rr", pred_col, label_all, args.outdir)

    # ---------- Track B: High-confidence gate + re-cal ----------
    label_g = f"{args.label}_gated_a{str(args.alpha_gated).replace('.','')}"
    print(f"\n[Track B] High-confidence :: gate (sqi>={args.thr_sqi}, snr>={args.thr_snr}), alpha={args.alpha_gated}")
    gate_mask = (df["sqi"] >= args.thr_sqi) & (df["psd_snr_db"] >= args.thr_snr)
    dfg = df[gate_mask].copy()
    print(f"[Track B] retained {len(dfg)}/{len(df)} = {len(dfg)/max(1,len(df)):.3f}")
    dfg = dfg.dropna(subset=["rr", pred_col, "zsnr"]).copy()
    dfg_cp = add_conformal_pi_mondrian(
        dfg, y_col="rr", yhat_col=pred_col,
        feature_col="zsnr", n_bins=args.n_bins, alpha=args.alpha_gated,
        n_folds=args.n_folds, group_col="subject"
    )
    out_parquet_b = os.path.join("data/processed", f"windows_with_pred_{label_g}.parquet")
    dfg_cp.to_parquet(out_parquet_b, index=False)
    # metrics
    base_metrics_b = basic_metrics(dfg_cp, "rr", pred_col)
    pi_m_b = pi_metrics(dfg_cp, "rr")
    pd.DataFrame([{
        **{"label": label_g, "mode": "high-confidence", "pred_col": pred_col,
           "gate_sqi": args.thr_sqi, "gate_snr": args.thr_snr,
           "retention": len(dfg_cp)/len(df)},
        **base_metrics_b, **pi_m_b
    }]).to_csv(os.path.join(args.outdir, f"{label_g}_metrics.csv"), index=False)
    print("High-conf (point):", base_metrics_b)
    print("High-conf (PI):   ", pi_m_b)
    width_tradeoff(dfg_cp, "rr", pred_col, label_g, args.outdir)

    # ---------- Side-by-side summary ----------
    summary = pd.DataFrame([
        {"track": "All-comers", **base_metrics_a, **pi_m_a, "retention": 1.0,
         "alpha": args.alpha_all, "bins": args.n_bins, "label": label_all},
        {"track": "High-confidence", **base_metrics_b, **pi_m_b,
         "retention": len(dfg_cp)/len(df),
         "alpha": args.alpha_gated, "bins": args.n_bins, "label": label_g,
         "gate_sqi": args.thr_sqi, "gate_snr": args.thr_snr},
    ])
    summary_path = os.path.join(args.outdir, f"{args.label}_two_track_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")

    # quick bar plots
    plt.figure()
    plt.bar(summary["track"], summary["coverage"])
    plt.ylim(0.0, 1.0)
    plt.title("Conformal Coverage")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.label}_two_track_coverage.png"), dpi=200)

    plt.figure()
    plt.bar(summary["track"], summary["mean_width"])
    plt.title("Conformal Mean Width")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"{args.label}_two_track_width.png"), dpi=200)

    print("\nDone. Artifacts under:")
    print(" - processed:", out_parquet_a, out_parquet_b)
    print(" - derived:", args.outdir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pick_rrhat_col(df: pd.DataFrame) -> str:
    # rr_hat 후보 자동 탐지 (fused_best 파이프라인은 rr_hat_freq)
    rrhat_cols = [c for c in df.columns if re.match(r"rr_hat", c)]
    if not rrhat_cols:
        # 백업: 과거 이름들 지원
        for c in ["rr_hat_freq", "rr_hat", "rr_pred"]:
            if c in df.columns:
                rrhat_cols = [c]; break
    assert rrhat_cols, f"No rr_hat* column found. columns={list(df.columns)[:20]}"
    return rrhat_cols[0]

def point_metrics(df: pd.DataFrame, label: str) -> dict:
    rrhat = pick_rrhat_col(df)
    sub = df.dropna(subset=["rr", rrhat]).copy()
    err = (sub[rrhat] - sub["rr"]).abs()
    out = {
        "label": label,
        "n": int(len(sub)),
        "MAE": float(err.mean()),
        "MdAE": float(err.median()),
        "RMSE": float(np.sqrt(((sub[rrhat]-sub["rr"])**2).mean())),
    }
    return out

def pi_metrics(df: pd.DataFrame, label: str) -> dict:
    sub = df.dropna(subset=["rr","pi_lo","pi_hi"]).copy()
    cov = ((sub["rr"]>=sub["pi_lo"])&(sub["rr"]<=sub["pi_hi"])).mean()
    width = (sub["pi_hi"]-sub["pi_lo"]).mean()
    out = {
        "label": label,
        "n": int(len(sub)),
        "coverage": float(cov),
        "mean_width": float(width),
    }
    return out

def per_subject_mae(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rrhat = pick_rrhat_col(df)
    sub = df.dropna(subset=["subject","rr", rrhat]).copy()
    g = sub.groupby("subject", as_index=False)
    out = g.apply(lambda gdf: pd.Series({
        "n": int(len(gdf)),
        "MAE": float((gdf[rrhat] - gdf["rr"]).abs().mean())
    }), include_groups=False)
    out["label"] = label
    return out

def overlay_width_tradeoff(in_csv_a: str, in_csv_b: str, out_png: str, labels=("all","gated")):
    a = pd.read_csv(in_csv_a)
    b = pd.read_csv(in_csv_b)
    plt.figure()
    plt.plot(a["retention"], a["coverage"], marker="o", label=labels[0])
    plt.plot(b["retention"], b["coverage"], marker="o", label=labels[1])
    plt.xlabel("Retention"); plt.ylabel("Coverage"); plt.grid(True)
    plt.legend()
    plt.title("Width-gating: Coverage vs Retention")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", required=True,
                    help="Track A parquet: e.g., data/processed/windows_with_pred_fused_best_all_a008.parquet")
    ap.add_argument("--gated", required=True,
                    help="Track B parquet: e.g., data/processed/windows_with_pred_fused_best_gated_a008.parquet")
    ap.add_argument("--label-all", default="all_a008")
    ap.add_argument("--label-gated", default="gated_a008")
    ap.add_argument("--outdir", default="data/derived")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df_all = pd.read_parquet(args.all)
    df_gt  = pd.read_parquet(args.gated)
    n_all = int(len(df_all.dropna(subset=["rr"])))
    n_gt  = int(len(df_gt.dropna(subset=["rr"])))
    retention = n_gt / n_all if n_all else np.nan

    rows = []
    rows.append(point_metrics(df_all, args.label_all) | pi_metrics(df_all, args.label_all))
    rows.append(point_metrics(df_gt,  args.label_gated) | pi_metrics(df_gt,  args.label_gated))
    topline = pd.DataFrame(rows)[["label","n","MAE","MdAE","RMSE","coverage","mean_width"]]
    topline["retention"] = [1.0, retention]  # all=1.0, gated≈0.648
    
    # 1) Topline 표 (point + PI)
    rows = []
    rows.append(point_metrics(df_all, args.label_all) | pi_metrics(df_all, args.label_all))
    rows.append(point_metrics(df_gt,  args.label_gated) | pi_metrics(df_gt,  args.label_gated))
    topline = pd.DataFrame(rows)[["label","n","MAE","MdAE","RMSE","coverage","mean_width"]]
    topline.to_csv(os.path.join(args.outdir,"two_track_topline.csv"), index=False)

    # 2) By-subject MAE 비교 표 + 그림
    s_all = per_subject_mae(df_all, args.label_all).rename(columns={"MAE":"MAE_all","n":"n_all"})
    s_gt  = per_subject_mae(df_gt,  args.label_gated).rename(columns={"MAE":"MAE_gated","n":"n_gated"})
    merged = pd.merge(s_all[["subject","MAE_all"]],
                      s_gt[["subject","MAE_gated"]],
                      on="subject", how="outer").fillna(np.nan)
    merged["delta"] = merged["MAE_gated"] - merged["MAE_all"]  # 음수면 gated가 더 좋음
    merged.sort_values("MAE_all", ascending=False, inplace=True)
    merged.to_csv(os.path.join(args.outdir,"two_track_by_subject.csv"), index=False)

    # 시각화(페어 바차트)
    plt.figure(figsize=(8,10))
    y = np.arange(len(merged))
    plt.barh(y+0.20, merged["MAE_all"], height=0.4, label=args.label_all)
    plt.barh(y-0.20, merged["MAE_gated"], height=0.4, label=args.label_gated)
    plt.yticks(y, merged["subject"].astype(int))
    plt.gca().invert_yaxis()
    plt.xlabel("MAE (bpm)"); plt.title("By-subject MAE: All vs Gated"); plt.grid(axis="x", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,"two_track_by_subject_mae.png"), dpi=220)

    # 3) Width-gating 곡선 오버레이(이미 앞서 저장한 CSV가 있으면 자동 사용)
    csv_a = os.path.join(args.outdir, f"{args.label_all}_width_tradeoff.csv")
    csv_b = os.path.join(args.outdir, f"{args.label_gated}_width_tradeoff.csv")
    if os.path.exists(csv_a) and os.path.exists(csv_b):
        overlay_width_tradeoff(csv_a, csv_b,
            os.path.join(args.outdir,"two_track_width_tradeoff_overlay.png"),
            labels=(args.label_all, args.label_gated))

    # 4) 콘솔 요약
    print("\n=== Topline ===")
    print(topline.round(3).to_string(index=False))
    print(f"\nSaved under: {args.outdir}")
    print(" - two_track_topline.csv")
    print(" - two_track_by_subject.csv, two_track_by_subject_mae.png")
    if os.path.exists(csv_a) and os.path.exists(csv_b):
        print(" - two_track_width_tradeoff_overlay.png")

if __name__ == "__main__":
    main()

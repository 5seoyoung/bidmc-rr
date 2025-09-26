#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RR 평가 스크립트
- Overall, Per-subject(Top-10 worst), SQI 게이트 스윕
- 최소 커버리지 제약(min_coverage) 하 최적 게이트 제안
- 라벨 여러 개(raw, riiv 등) 동시 비교 + 그림/CSV 저장
"""

import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def metrics(df: pd.DataFrame, yhat_col="rr_hat_freq", y_col="rr") -> dict:
    d = df[[yhat_col, y_col]].dropna()
    if d.empty:
        return {"n": 0, "MAE": np.nan, "MdAE": np.nan, "RMSE": np.nan}
    err = np.abs(d[yhat_col].to_numpy() - d[y_col].to_numpy())
    mae = float(np.mean(err))
    mdae = float(np.median(err))
    rmse = float(np.sqrt(np.mean((d[yhat_col] - d[y_col]) ** 2)))
    return {"n": int(len(d)), "MAE": mae, "MdAE": mdae, "RMSE": rmse}


def per_subject_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("subject"):
        m = metrics(g)
        rows.append({"subject": int(s), "n": m["n"], "MAE": m["MAE"], "MdAE": m["MdAE"], "RMSE": m["RMSE"]})
    per = pd.DataFrame(rows).sort_values("MAE", ascending=False)
    for c in ["MAE","MdAE","RMSE"]:
        per[c] = per[c].round(3)
    per["n"] = per["n"].astype(int)
    return per


def sweep_sqi_gate(df: pd.DataFrame,
                   mode: str,
                   grid_q=np.linspace(0.0, 1.0, 21)) -> pd.DataFrame:
    """
    SQI 게이트 스윕:
      - mode: "low_good" -> keep sqi <= thr
              "high_good"-> keep sqi >= thr
    grid_q: 임계값 후보를 sqi의 분위수로 생성
    """
    sqi = df["sqi"].dropna()
    if sqi.empty:
        return pd.DataFrame(columns=["n","MAE","MdAE","RMSE","coverage","thr","mode"])

    thrs = np.quantile(sqi.to_numpy(), grid_q)
    rows = []
    for thr in thrs:
        if mode == "low_good":
            sub = df[df["sqi"] <= thr]
        else:
            sub = df[df["sqi"] >= thr]
        m = metrics(sub)
        rows.append({
            "n": m["n"], "MAE": m["MAE"], "MdAE": m["MdAE"], "RMSE": m["RMSE"],
            "coverage": float(len(sub) / len(df)),
            "thr": float(thr),
            "mode": "sqi<=thr (low is good)" if mode == "low_good" else "sqi>=thr (high is good)"
        })
    res = pd.DataFrame(rows)
    # 정렬은 coverage 증가 순서로
    return res.sort_values("coverage").reset_index(drop=True)


def choose_gate_under_constraint(sweep_df: pd.DataFrame, min_coverage: float) -> dict | None:
    """
    최소 커버리지 제약 하에서 MAE 최소 임계치 선택.
    """
    cand = sweep_df[sweep_df["coverage"] >= float(min_coverage)].copy()
    if cand.empty:
        return None
    i = cand["MAE"].idxmin()
    row = cand.loc[i]
    return {
        "n": int(row["n"]),
        "MAE": float(row["MAE"]),
        "MdAE": float(row["MdAE"]),
        "RMSE": float(row["RMSE"]),
        "coverage": float(row["coverage"]),
        "thr": float(row["thr"]),
        "mode": str(row["mode"]),
    }


def plot_sweep(sweep_low: pd.DataFrame, sweep_high: pd.DataFrame, title: str, out_png: Path):
    plt.figure(figsize=(7,5))
    plt.plot(sweep_low["coverage"], sweep_low["MAE"], label="low-is-good: sqi<=thr")
    plt.plot(sweep_high["coverage"], sweep_high["MAE"], label="high-is-good: sqi>=thr")
    plt.xlabel("Coverage")
    plt.ylabel("MAE (bpm)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_per_subject(per: pd.DataFrame, title: str, out_png: Path, topk: int = 10):
    worst = per.head(topk).iloc[::-1]  # 바 차트 위에서 아래로
    plt.figure(figsize=(6, 4 + 0.2*topk))
    plt.barh([f"{int(s)}" for s in worst["subject"]], worst["MAE"])
    plt.xlabel("MAE (bpm)")
    plt.ylabel("Subject")
    plt.title(title)
    for i, v in enumerate(worst["MAE"]):
        plt.text(v, i, f" {v:.1f}", va="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", required=True, help="평가할 parquet 파일들")
    ap.add_argument("--labels", nargs="+", required=True, help="각 파일 라벨 (paths와 길이 동일)")
    ap.add_argument("--min-coverage", type=float, default=0.60, help="추천 게이트 최소 커버리지")
    ap.add_argument("--outdir", type=str, default="data/derived")
    args = ap.parse_args()

    assert len(args.paths) == len(args.labels), "paths/labels 길이가 같아야 합니다."
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    for path, label in zip(args.paths, args.labels):
        df = pd.read_parquet(path)
        base_cols = {"subject", "t_start", "t_end", "sqi", "rr", "rr_hat_freq"}
        if not base_cols.issubset(df.columns):
            missing = base_cols - set(df.columns)
            raise ValueError(f"{path} missing columns: {missing}")

        # 전체/요약
        overall = metrics(df)
        print(f"\n=== {label.upper()} Overall (no gating) ===")
        print({k: (round(v,3) if isinstance(v, float) else v) for k, v in overall.items()})

        # 퍼서브젝트
        per = per_subject_table(df)
        print(f"\n=== {label.upper()} By subject (top 10 worst MAE) ===")
        print(per.head(10).to_string(index=False))

        # 저장
        per_csv = outdir / f"{label}_per_subject.csv"
        per.to_csv(per_csv, index=False)

        # 그래프: per-subject worst 10
        plot_per_subject(per, title=f"{label.upper()} — Worst 10 by MAE", out_png=outdir / f"{label}_per_subject_top10.png")

        # SQI 스윕 (두 가지 가정)
        sweep_low  = sweep_sqi_gate(df, mode="low_good")
        sweep_high = sweep_sqi_gate(df, mode="high_good")

        # 상위 5개 프린트(참고)
        best5_low  = sweep_low.nsmallest(5, "MAE")
        best5_high = sweep_high.nsmallest(5, "MAE")
        print(f"\n=== {label.upper()} SQI gating sweep — best 5 (assume low-is-good) ===")
        print(best5_low.assign(MAE=lambda d: d["MAE"].round(3),
                               MdAE=lambda d: d["MdAE"].round(3),
                               RMSE=lambda d: d["RMSE"].round(3),
                               coverage=lambda d: d["coverage"].round(2),
                               thr=lambda d: d["thr"].round(3)).to_string(index=False))
        print(f"\n=== {label.upper()} SQI gating sweep — best 5 (assume high-is-good) ===")
        print(best5_high.assign(MAE=lambda d: d["MAE"].round(3),
                                MdAE=lambda d: d["MdAE"].round(3),
                                RMSE=lambda d: d["RMSE"].round(3),
                                coverage=lambda d: d["coverage"].round(2),
                                thr=lambda d: d["thr"].round(3)).to_string(index=False))

        # 추천 게이트(커버리지 제약)
        best_low  = choose_gate_under_constraint(sweep_low, args.min_coverage)
        best_high = choose_gate_under_constraint(sweep_high, args.min_coverage)

        # 두 모드 중 MAE가 더 낮은 것을 채택
        choices = [b for b in [best_low, best_high] if b is not None]
        if choices:
            chosen = min(choices, key=lambda d: d["MAE"])
            print(f"\n>>> Suggested gate (min coverage >= {args.min_coverage:.2f}):")
            print({k: (round(v,3) if isinstance(v, float) else v) for k, v in chosen.items()})
        else:
            print(f"\n>>> Suggested gate: not available (cannot meet coverage >= {args.min_coverage:.2f})")
            chosen = None

        # 스윕 저장 & 플롯
        sweep_low.to_csv(outdir / f"{label}_sweep_low_is_good.csv", index=False)
        sweep_high.to_csv(outdir / f"{label}_sweep_high_is_good.csv", index=False)
        plot_sweep(sweep_low, sweep_high, title=f"{label.upper()} — SQI gating sweep", out_png=outdir / f"{label}_sweep.png")

        # 전체 결과 json 저장
        summary = {
            "label": label,
            "overall": overall,
            "suggested_gate": chosen,
        }
        with open(outdir / f"{label}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nSaved figures & sweeps under: {outdir}")


if __name__ == "__main__":
    main()

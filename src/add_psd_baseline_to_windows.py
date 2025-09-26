#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BIDMC 윈도우별 RR(PSD 피크) 베이스라인 예측 추가 스크립트
- CSV 모드(bidmc_csv) 또는 subject_XX 디렉토리 모드 지원
- RAW(원신호) / RIIV(엔벨로프) 모두 지원
- RIIV 기본 파라미터 상한 확장: lp=0.8 Hz, PSD 대역=(0.08, 0.8)로 고RR 커버

출력 컬럼:
  rr_hat_freq, psd_peak_hz, psd_peak_power, psd_snr_db
"""

import os, glob, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, sosfiltfilt, hilbert

PPG_COL_KEYS = ("pleth", "ppg", "photopleth")


# ---------------------------
# CSV 모드 유틸
# ---------------------------
def find_signals_csv(csv_root: Path, subject: int) -> Path:
    """
    bidmc_csv 내부에서 *_Signals.csv 파일을 subject 번호로 찾음.
    """
    if not csv_root or not csv_root.exists():
        raise FileNotFoundError(f"csv_root not found: {csv_root}")
    pattern = re.compile(rf".*_{subject:02d}_Signals\.csv$", re.IGNORECASE)
    candidates = [p for p in csv_root.rglob("*_Signals.csv") if pattern.match(p.name)]
    if not candidates:
        raise FileNotFoundError(f"No *_Signals.csv for subject {subject:02d} under: {csv_root}")
    return sorted(candidates)[0]


def _normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def read_ppg_from_signals_csv(path: Path) -> tuple[np.ndarray, str]:
    """
    *_Signals.csv에서 PPG(pleth) 컬럼을 찾아 1열만 로드.
    """
    hdr = pd.read_csv(path, nrows=0)
    cols = list(hdr.columns)
    col_norm = [_normalize_name(c) for c in cols]

    ppg_idx = None
    for i, cn in enumerate(col_norm):
        if any(k in cn for k in PPG_COL_KEYS):
            ppg_idx = i
            break
    if ppg_idx is None:
        # 컬럼명이 특이할 수 있으니 힌트를 출력
        raise ValueError(f"PPG-like column not found in {path.name}; first columns={cols[:8]}")

    ppg_col = cols[ppg_idx]
    ser = pd.read_csv(path, usecols=[ppg_col])[ppg_col]
    x = pd.to_numeric(ser, errors="coerce").astype(float).to_numpy()
    if np.isnan(x).all():
        x = np.zeros_like(x, dtype=float)
    else:
        x = np.nan_to_num(x, nan=float(np.nanmean(x)))
    return x, ppg_col


# ---------------------------
# subject_XX 모드 유틸 (폴백)
# ---------------------------
SKIP_FILE_TOKENS = [
    "breath", "numeric", "annotation", "alarm", "event",
    "spo2", "oxim", "hr_", "heartrate", "rr_", "resp_", "respiration",
    "abp", "bp_", "art", "arterial"
]

def list_candidate_csvs(subj_dir: str) -> list[str]:
    csvs = glob.glob(os.path.join(subj_dir, "**", "*.csv"), recursive=True)
    kept = []
    for p in csvs:
        name = os.path.basename(p).lower()
        if any(tok in name for tok in SKIP_FILE_TOKENS):
            continue
        kept.append(p)
    kept.sort(key=lambda x: os.path.getsize(x), reverse=True)
    return kept


def find_ppg_column_in_header(cols: list[str], col_hints: list[str]) -> list[str]:
    low = [c.lower() for c in cols]
    hits = []
    for i, c in enumerate(low):
        if any(h in c for h in col_hints):
            hits.append(cols[i])
    return hits


def choose_best_by_variance(path: str, cols: list[str]) -> str:
    variances = []
    for c in cols:
        try:
            x = pd.read_csv(path, usecols=[c])[c]
            x = pd.to_numeric(x, errors="coerce")
            variances.append(float(np.nanvar(x)))
        except Exception:
            variances.append(-np.inf)
    best_idx = int(np.nanargmax(variances))
    return cols[best_idx]


def try_load_column(path: str, col: str) -> np.ndarray:
    ser = pd.read_csv(path, usecols=[col])[col]
    x = pd.to_numeric(ser, errors="coerce").astype(float).to_numpy()
    if np.isnan(x).all():
        x = np.zeros_like(x, dtype=float)
    else:
        x = np.nan_to_num(x, nan=float(np.nanmean(x)))
    return x


def load_ppg_from_subject(subj_dir: str,
                          file_hints: list[str],
                          col_hints: list[str]) -> tuple[np.ndarray, str, str]:
    candidates = list_candidate_csvs(subj_dir)
    if not candidates:
        raise FileNotFoundError(f"No CSV files under: {subj_dir}")

    prioritized = [p for p in candidates if any(h in os.path.basename(p).lower() for h in file_hints)]
    search_order = prioritized + [p for p in candidates if p not in prioritized]

    for path in search_order:
        try:
            hdr = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        cols = list(hdr.columns)
        if not cols:
            continue
        hit_cols = find_ppg_column_in_header(cols, col_hints)
        if not hit_cols:
            continue
        col = choose_best_by_variance(path, hit_cols) if len(hit_cols) > 1 else hit_cols[0]
        try:
            x = try_load_column(path, col)
            if np.allclose(x, 0.0) or len(x) < 100:
                continue
            return x.astype(float), path, col
        except Exception:
            continue

    raise FileNotFoundError(
        f"Could not locate a PPG-like column in CSVs under: {subj_dir}\n"
        f"Tips: try --ppg-file-hints 'waveform signals' or --ppg-col-hints 'pleth ppg'"
    )


# ---------------------------
# 신호처리
# ---------------------------
def butter_sos_bandpass(low, high, fs, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sos

def butter_sos_lowpass(cut, fs, order=4):
    sos = butter(order, cut, btype="lowpass", fs=fs, output="sos")
    return sos

def riiv_envelope(x: np.ndarray,
                  fs: float,
                  bp=(0.8, 3.0),
                  lp=0.8,
                  decim: int = 25) -> tuple[np.ndarray, float]:
    """
    RIIV(Respiratory-Induced Intensity/Amplitude Variations) 추출:
      1) 심박 대역 BP(0.8–3.0 Hz)
      2) Hilbert 엔벨로프
      3) LPF(lr<=0.8 Hz)
      4) 필요시 디시메이션
    """
    if x is None or len(x) < int(fs * 4):
        return np.array([], dtype=float), 1.0

    x = x.astype(float)
    x = x - float(np.nanmean(x))
    if np.allclose(np.nanstd(x), 0.0):
        return np.array([], dtype=float), 1.0

    # 1) BP
    sos_bp = butter_sos_bandpass(bp[0], bp[1], fs, order=4)
    xb = sosfiltfilt(sos_bp, x)

    # 2) envelope
    env = np.abs(hilbert(xb))

    # 3) LP
    sos_lp = butter_sos_lowpass(lp, fs, order=4)
    env = sosfiltfilt(sos_lp, env)

    # 4) decimation (정수배만 허용)
    if decim and decim > 1:
        n = len(env) // decim
        if n <= 1:
            return env, fs
        env = env[: n * decim].reshape(n, decim).mean(axis=1)  # 간단 평균 다운샘플
        fs_out = fs / decim
    else:
        fs_out = fs

    # DC 제거(소프트 detrend)
    env = env - float(np.nanmedian(env))
    return env.astype(float), float(fs_out)


def psd_peak_rr(x: np.ndarray, fs: float, band=(0.1, 0.6), nperseg=2048):
    """
    Welch PSD에서 호흡대역 최고 피크를 RR(bpm)으로 환산.
    반환: (rr_bpm, peak_freq, peak_power, snr_db)
    """
    if x is None or len(x) < int(fs * 4):
        return np.nan, np.nan, np.nan, np.nan
    x = x - float(np.nanmean(x))
    if np.allclose(np.nanstd(x), 0.0):
        return np.nan, np.nan, np.nan, np.nan

    f, Pxx = welch(x, fs=fs, nperseg=min(int(nperseg), len(x)))
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return np.nan, np.nan, np.nan, np.nan

    fb, Pb = f[m], Pxx[m]
    i = int(np.nanargmax(Pb))
    peak_f = float(fb[i])
    peak_p = float(Pb[i])

    # 간단 SNR: 피크 주변 제외한 대역 중앙값 대비
    if len(fb) > 1:
        half_bw_bins = max(1, int(0.02 / (fb[1] - fb[0])))
    else:
        half_bw_bins = 1
    mask = np.ones_like(Pb, dtype=bool)
    mask[max(0, i - half_bw_bins):min(len(Pb), i + half_bw_bins + 1)] = False
    noise = float(np.nanmedian(Pb[mask])) if np.any(mask) else float(np.nanmedian(Pb))
    snr_db = 10.0 * np.log10(peak_p / noise) if noise > 0 else np.nan

    rr_bpm = 60.0 * peak_f
    return rr_bpm, peak_f, peak_p, snr_db


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data/processed/windows.parquet")
    ap.add_argument("--raw-root", default="data/raw/bidmc", help="subject_XX 폴더 루트(옵션)")
    ap.add_argument("--csv-root", type=str, default=None,
                    help="BIDMC 'bidmc_csv' 폴더 경로 (권장)")
    ap.add_argument("--out", default="data/processed/windows_with_pred.parquet")

    ap.add_argument("--method", choices=["raw", "riiv"], default="raw",
                    help="RAW 원신호 vs RIIV 엔벨로프 기반")
    ap.add_argument("--fs", type=float, default=125.0, help="원시 신호 샘플링레이트(Hz)")

    # RAW용 PSD 탐색 대역
    ap.add_argument("--raw-band", type=float, nargs=2, default=[0.1, 0.6],
                    help="RAW PSD 탐색대역 (Hz)")

    # RIIV 파이프라인 파라미터
    ap.add_argument("--bp", type=float, nargs=2, default=[0.8, 3.0],
                    help="RIIV: PPG 대역통과(Hz) [기본 0.8–3.0]")
    ap.add_argument("--lp", type=float, default=0.8,
                    help="RIIV: 엔벨로프 저역통과 컷오프(Hz) [기본 0.8]")
    ap.add_argument("--decim", type=int, default=25,
                    help="RIIV: 디시메이션 팩터 (fs/decim) [기본 25 → 5 Hz]")
    ap.add_argument("--riiv-band", type=float, nargs=2, default=[0.08, 0.8],
                    help="RIIV PSD 탐색대역 (Hz) [기본 0.08–0.8]")

    # 휴리스틱 파일/컬럼 힌트(폴백 모드용)
    ap.add_argument("--ppg-file-hints", type=str, default="ppg pleth wave waveform signals",
                    help="파일명 우선 힌트(공백 구분)")
    ap.add_argument("--ppg-col-hints", type=str, default="ppg pleth",
                    help="컬럼명 우선 힌트(공백 구분)")

    args = ap.parse_args()

    df = pd.read_parquet(args.windows)
    need = {"subject", "t_start", "t_end"}
    if not need.issubset(df.columns):
        raise ValueError(f"{args.windows} must contain columns: {need}")

    out = df.copy()
    out["rr_hat_freq"] = np.nan
    out["psd_peak_hz"] = np.nan
    out["psd_peak_power"] = np.nan
    out["psd_snr_db"] = np.nan

    file_hints = [s.strip().lower() for s in args.ppg_file_hints.split() if s.strip()]
    col_hints  = [s.strip().lower() for s in args.ppg_col_hints.split() if s.strip()]

    fs_raw = float(args.fs)
    band_raw = tuple(args.raw_band)
    bp = tuple(args.bp)
    lp = float(args.lp)
    decim = int(args.decim)
    band_riiv = tuple(args.riiv_band)

    csv_root = Path(args.csv_root) if args.csv_root else None
    use_csv_mode = csv_root is not None and csv_root.exists()

    if use_csv_mode:
        print(f"[MODE] CSV mode: {csv_root}")
    else:
        print(f"[MODE] subject_XX mode: {args.raw_root}")

    print(f"[METHOD] {args.method.upper()}")

    for subj, g in out.groupby("subject"):
        subj = int(subj)
        # --- PPG 로딩 ---
        if use_csv_mode:
            signals_csv = find_signals_csv(csv_root, subj)
            ppg, used_col = read_ppg_from_signals_csv(signals_csv)
            used_file = str(signals_csv)
        else:
            subj_dir = os.path.join(args.raw_root, f"subject_{subj:02d}")
            ppg, used_file, used_col = load_ppg_from_subject(subj_dir, file_hints, col_hints)

        print(f"✓ subject {subj:02d} PPG -> {os.path.basename(used_file)} :: '{used_col}' (len={len(ppg)})")

        # --- 윈도우 루프 ---
        for idx, row in g.iterrows():
            s = int(round(row["t_start"] * fs_raw))
            e = int(round(row["t_end"]   * fs_raw))
            if not (0 <= s < e <= len(ppg)):
                rr = pf = pp = snr = np.nan
            else:
                seg = ppg[s:e]
                if args.method == "riiv":
                    env, fs_env = riiv_envelope(seg, fs=fs_raw, bp=bp, lp=lp, decim=decim)
                    rr, pf, pp, snr = psd_peak_rr(env, fs=fs_env, band=band_riiv, nperseg=256)
                else:
                    rr, pf, pp, snr = psd_peak_rr(seg, fs=fs_raw, band=band_raw, nperseg=2048)

            out.at[idx, "rr_hat_freq"]   = rr
            out.at[idx, "psd_peak_hz"]   = pf
            out.at[idx, "psd_peak_power"]= pp
            out.at[idx, "psd_snr_db"]    = snr

        print(f"✓ subject {subj:02d} pred done")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

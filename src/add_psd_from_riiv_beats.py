#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, sosfiltfilt, find_peaks

PPG_COL_KEYS = ("pleth", "ppg", "photopleth")

def find_signals_csv(csv_root: Path, subject: int) -> Path:
    pattern = re.compile(rf".*_{subject:02d}_Signals\.csv$", re.IGNORECASE)
    cands = [p for p in csv_root.rglob("*_Signals.csv") if pattern.match(p.name)]
    if not cands:
        raise FileNotFoundError(f"No *_Signals.csv for subject {subject:02d} under: {csv_root}")
    return sorted(cands)[0]

def read_ppg_from_signals_csv(path: Path):
    df = pd.read_csv(path)
    ppg_col = None
    for c in df.columns:
        if any(k in c.lower() for k in PPG_COL_KEYS):
            ppg_col = c
            break
    if ppg_col is None:
        raise ValueError(f"PPG column not found in {path.name}; columns={list(df.columns)[:10]}...")
    x = pd.to_numeric(df[ppg_col], errors="coerce").astype(float).to_numpy()
    x = np.nan_to_num(x, nan=np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0)
    return x, ppg_col

def bandpass(x, fs, lo, hi, order=4):
    lo = max(1e-3, lo) / (fs / 2.0)
    hi = min(fs / 2.0 - 1e-3, hi) / (fs / 2.0)
    sos = butter(order, [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)

def psd_peak_rr(x: np.ndarray, fs: float, band=(0.1, 0.6), nperseg=None):
    """Welch PSD에서 호흡대역 피크 → RR(bpm), 간단 SNR(dB) 계산"""
    if x is None or len(x) < max(16, int(fs * 4)):
        return (np.nan, np.nan, np.nan, np.nan)
    x = x - np.nanmean(x)
    if np.allclose(np.nanstd(x), 0):
        return (np.nan, np.nan, np.nan, np.nan)
    if nperseg is None:
        nperseg = min(len(x), 256)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m):
        return (np.nan, np.nan, np.nan, np.nan)
    fb, Pb = f[m], Pxx[m]
    i = int(np.nanargmax(Pb))
    peak_f = float(fb[i]); peak_p = float(Pb[i])
    # 주변 띠 제외하고 중앙값을 노이즈로
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

def make_riiv_from_beats(ppg: np.ndarray, fs: float, fs_riiv: float = 4.0,
                         min_peak_distance_s: float = 0.3):
    """
    1) 심박 대역 통과(0.3–8 Hz) → 2) 피크 검출 → 3) 각 비트에서 (peak - 지역 최소) 진폭
    → 4) (t_beats, amp) 시퀀스를 균일 그리드(4 Hz)로 보간 → RIIV 시계열 반환
    """
    # 1) 심박 대역 필터
    x = bandpass(ppg, fs, 0.3, 8.0)
    # 2) 피크 검출
    min_dist = max(1, int(fs * min_peak_distance_s))
    prom = 0.2 * np.nanstd(x)  # 완전 보수적
    peaks, _ = find_peaks(x, distance=min_dist, prominence=max(1e-6, prom))
    if len(peaks) < 5:
        # 비트가 너무 적으면 빈 RIIV 반환
        T = len(ppg) / fs
        t_u = np.arange(0, T, 1.0 / fs_riiv)
        return np.full_like(t_u, fill_value=np.nan, dtype=float), t_u
    # 3) 각 비트의 peak-to-trough 진폭
    troughs = []
    amps = []
    times = []
    for i in range(1, len(peaks)):
        a, b = peaks[i - 1], peaks[i]
        if b <= a + 1:  # 이상 케이스
            continue
        seg = x[a:b]
        tr_rel = int(np.argmin(seg))
        tr = a + tr_rel
        amp = x[b] - x[tr]
        if np.isfinite(amp):
            amps.append(float(amp))
            times.append(float(b / fs))
            troughs.append(tr)
    amps = np.array(amps, dtype=float)
    times = np.array(times, dtype=float)
    if len(amps) < 3:
        T = len(ppg) / fs
        t_u = np.arange(0, T, 1.0 / fs_riiv)
        return np.full_like(t_u, fill_value=np.nan, dtype=float), t_u
    # 4) 균일 재샘플
    T = len(ppg) / fs
    t_u = np.arange(0, T, 1.0 / fs_riiv)
    # 경계 처리
    t0, t1 = times[0], times[-1]
    v0, v1 = amps[0], amps[-1]
    # 보간
    riiv = np.interp(t_u, times, amps, left=v0, right=v1)
    # 안정화를 위해 평균/표준편차 정규화(선택적)
    if np.nanstd(riiv) > 0:
        riiv = (riiv - np.nanmean(riiv)) / np.nanstd(riiv)
    return riiv.astype(float), t_u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data/processed/windows.parquet")
    ap.add_argument("--csv-root", type=str, required=True,
                    help="PhysioNet BIDMC 'bidmc_csv' 폴더 경로")
    ap.add_argument("--out", default="data/processed/windows_with_pred_riiv_beats.parquet")
    ap.add_argument("--fs", type=float, default=125.0)
    ap.add_argument("--fs-riiv", type=float, default=4.0)
    ap.add_argument("--band", type=float, nargs=2, default=[0.1, 0.6])
    ap.add_argument("--min-peak-distance", type=float, default=0.3,
                    help="PPG 피크 최소 간격(초)")
    args = ap.parse_args()

    csv_root = Path(args.csv_root)
    dfw = pd.read_parquet(args.windows)
    need = {"subject", "t_start", "t_end"}
    if not need.issubset(dfw.columns):
        raise ValueError(f"{args.windows} must contain {need}")

    out = dfw.copy()
    out["rr_hat_freq"] = np.nan
    out["psd_peak_hz"] = np.nan
    out["psd_peak_power"] = np.nan
    out["psd_snr_db"] = np.nan

    fs = float(args.fs)
    fs_r = float(args.fs_riiv)
    band = tuple(args.band)

    print(f"[MODE] RIIV-from-beats @ {fs_r} Hz, band={band}")
    for subj, g in out.groupby("subject"):
        sig_csv = find_signals_csv(csv_root, int(subj))
        ppg, used_col = read_ppg_from_signals_csv(sig_csv)
        print(f"✓ subject {int(subj):02d} PPG -> {sig_csv.name} :: '{used_col}' (len={len(ppg)})")
        riiv, t_u = make_riiv_from_beats(ppg, fs=fs, fs_riiv=fs_r,
                                         min_peak_distance_s=args.min_peak_distance)
        N_r = len(riiv)

        for idx, row in g.iterrows():
            s = int(round(row["t_start"] * fs_r))
            e = int(round(row["t_end"]   * fs_r))
            if not (0 <= s < e <= N_r):
                rr = pf = pp = snr = np.nan
            else:
                seg = riiv[s:e]
                rr, pf, pp, snr = psd_peak_rr(seg, fs=fs_r, band=band,
                                              nperseg=min(len(seg), 256))
            out.at[idx, "rr_hat_freq"]   = rr
            out.at[idx, "psd_peak_hz"]   = pf
            out.at[idx, "psd_peak_power"]= pp
            out.at[idx, "psd_snr_db"]    = snr

        print(f"✓ subject {int(subj):02d} RIIV-beats pred done")

    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()

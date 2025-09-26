#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, welch

PPG_KEYS = ("pleth", "ppg", "photopleth")

# ----------------------------- I/O helpers -----------------------------
def find_signals_csv(csv_root: Path, subject: int) -> Path:
    pat = re.compile(rf".*_{subject:02d}_Signals\.csv$", re.IGNORECASE)
    cands = [p for p in csv_root.rglob("*_Signals.csv") if pat.match(p.name)]
    if not cands:
        raise FileNotFoundError(f"No *_Signals.csv for subject {subject:02d} under: {csv_root}")
    return sorted(cands)[0]

def read_ppg(path: Path):
    df = pd.read_csv(path)
    col = None
    for c in df.columns:
        if any(k in c.lower() for k in PPG_KEYS):
            col = c; break
    if col is None:
        raise ValueError(f"PPG column not found in {path.name}; columns={list(df.columns)[:10]}")
    x = pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()
    x = np.nan_to_num(x, nan=np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0)
    return x, col

# ----------------------------- DSP utils ------------------------------
def bandpass(x, fs, lo, hi, order=4):
    lo = max(1e-3, lo) / (fs/2.0); hi = min(fs/2.0-1e-3, hi) / (fs/2.0)
    sos = butter(order, [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)

def lowpass(x, fs, fc, order=4):
    wn = min(fs/2.0-1e-3, fc) / (fs/2.0)
    sos = butter(order, wn, btype="lowpass", output="sos")
    return sosfiltfilt(sos, x)

def resample_uniform(t_src, v_src, fs_out, T_total):
    # 균일 타임스탬프(0..T_total)로 선형 보간
    t_u = np.arange(0.0, T_total, 1.0/fs_out)
    if len(t_src) < 2:
        return np.full_like(t_u, np.nan, dtype=float), t_u
    v = np.interp(t_u, t_src, v_src, left=v_src[0], right=v_src[-1])
    return v.astype(float), t_u

def psd_rr_and_snr(x, fs, band=(0.1,0.6), exclude_hz=0.04, nperseg=None):
    """Welch PSD에서 호흡대역 피크→RR, 피크±exclude_hz 제외 중앙값으로 SNR(dB)."""
    if x is None or len(x) < max(16, int(fs*4)) or np.allclose(np.nanstd(x), 0):
        return (np.nan, np.nan, np.nan, np.nan)
    x = x - np.nanmean(x)
    if nperseg is None: nperseg = min(len(x), 256)
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    m = (f >= band[0]) & (f <= band[1])
    if not np.any(m): return (np.nan, np.nan, np.nan, np.nan)
    fb, Pb = f[m], Pxx[m]
    i = int(np.nanargmax(Pb))
    fpk, Ppk = float(fb[i]), float(Pb[i])

    # exclude ±exclude_hz around fpk
    mask = np.ones_like(Pb, dtype=bool)
    mask &= ~((fb >= fpk - exclude_hz) & (fb <= fpk + exclude_hz))
    noise = float(np.nanmedian(Pb[mask])) if np.any(mask) else float(np.nanmedian(Pb))
    snr_db = 10.0*np.log10(Ppk/noise) if noise > 0 else np.nan
    rr_bpm = 60.0*fpk
    return rr_bpm, fpk, Ppk, snr_db

# ----------------------------- Feature tracks -------------------------
def hilbert_riiv(ppg, fs):
    """심박대역(0.5–8Hz) → Hilbert amplitude → 호흡대역(0.1–0.6Hz)"""
    x = bandpass(ppg, fs, 0.5, 8.0)
    env = np.abs(hilbert(x))
    env = (env - np.nanmean(env)) / (np.nanstd(env) + 1e-12)
    riiv = bandpass(env, fs, 0.1, 0.6)
    return riiv

def detect_peaks(ppg, fs):
    x = bandpass(ppg, fs, 0.5, 8.0)
    prom = 0.2*np.nanstd(x)
    dist = max(1, int(0.3*fs))  # 0.3s
    peaks, _ = find_peaks(x, distance=dist, prominence=max(1e-6, prom))
    return peaks, x

def prv_series(ppg, fs, fs_out=4.0):
    """피크→IBI(초)→시간축에 맞춘 HRV(또는 IBI) 균일 재샘플."""
    peaks, x = detect_peaks(ppg, fs)
    if len(peaks) < 4:
        T = len(ppg)/fs
        t_u = np.arange(0, T, 1.0/fs_out)
        return np.full_like(t_u, np.nan, dtype=float), t_u
    t_peaks = peaks/fs
    ibi = np.diff(t_peaks)  # seconds
    t_mid = (t_peaks[1:] + t_peaks[:-1]) / 2.0
    # IBI를 쓰거나, HR = 60/IBI를 써도 됨. 여기서는 IBI 변동으로 진행.
    T_total = len(ppg)/fs
    v_u, t_u = resample_uniform(t_mid, ibi, fs_out, T_total)
    # 호흡대역 통과(0.1–0.6Hz) 위해 정규화
    if np.nanstd(v_u) > 0: v_u = (v_u - np.nanmean(v_u))/np.nanstd(v_u)
    v_u = bandpass(np.nan_to_num(v_u, nan=0.0), fs_out, 0.1, 0.6)
    return v_u, t_u

# ----------------------------- Main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="data/processed/windows.parquet")
    ap.add_argument("--csv-root", required=True, type=str)
    ap.add_argument("--out-base", default="data/processed/windows_with_pred")
    ap.add_argument("--fs", type=float, default=125.0)
    ap.add_argument("--fs-uni", type=float, default=4.0, help="PRV/HILBERT 공통 균일 샘플링")
    ap.add_argument("--resp-band", type=float, nargs=2, default=[0.1,0.6])
    args = ap.parse_args()

    dfw = pd.read_parquet(args.windows)
    need = {"subject","t_start","t_end","sqi","rr"}
    if not need.issubset(dfw.columns):
        raise ValueError(f"{args.windows} must contain {need}")

    fs = float(args.fs); fsu = float(args.fs_uni); band = tuple(args.resp_band)
    csv_root = Path(args.csv_root)

    # 결과 담을 복사본들
    base_cols = list(dfw.columns)
    H = dfw.copy(); P = dfw.copy(); F = dfw.copy()
    for df in (H,P,F):
        for c in ("rr_hat_freq","psd_peak_hz","psd_peak_power","psd_snr_db"):
            df[c] = np.nan

    for subj, g_idx in dfw.groupby("subject").groups.items():
        sig_csv = find_signals_csv(csv_root, int(subj))
        ppg, col = read_ppg(sig_csv)
        T = len(ppg)/fs

        # 트랙 계산
        riiv_h = hilbert_riiv(ppg, fs)
        prv_u, t_u = prv_series(ppg, fs, fs_out=fsu)

        # 윈도우별 추정
        for idx in g_idx:
            t0 = dfw.at[idx,"t_start"]; t1 = dfw.at[idx,"t_end"]
            # RAW는 기존 파일이 있으니 여기서는 H/P/F만 만들되, 동일 창 길이 적용
            # Hilbert: 원 샘플링(fs)
            s = int(round(t0*fs)); e = int(round(t1*fs))
            if 0 <= s < e <= len(riiv_h):
                rr, fpk, ppk, snr = psd_rr_and_snr(riiv_h[s:e], fs, band, exclude_hz=0.04)
                H.at[idx,"rr_hat_freq"]=rr; H.at[idx,"psd_peak_hz"]=fpk
                H.at[idx,"psd_peak_power"]=ppk; H.at[idx,"psd_snr_db"]=snr

            # PRV: 균일  fsu
            s2 = int(round(t0*fsu)); e2 = int(round(t1*fsu))
            if prv_u is not None and 0 <= s2 < e2 <= len(prv_u):
                rr, fpk, ppk, snr = psd_rr_and_snr(prv_u[s2:e2], fsu, band, exclude_hz=0.04)
                P.at[idx,"rr_hat_freq"]=rr; P.at[idx,"psd_peak_hz"]=fpk
                P.at[idx,"psd_peak_power"]=ppk; P.at[idx,"psd_snr_db"]=snr

        # Fused: 창별로 H vs P 중 SNR 높은 것을 채택(동률 시 H 우선)
        for idx in g_idx:
            cand = []
            for src, df in (("H",H),("P",P)):
                rr = df.at[idx,"rr_hat_freq"]; snr = df.at[idx,"psd_snr_db"]
                if np.isfinite(rr) and np.isfinite(snr): cand.append((snr, rr, src))
            if cand:
                cand.sort(reverse=True)  # snr 내림차순
                snr, rr, src = cand[0]
                F.at[idx,"rr_hat_freq"]=rr
                F.at[idx,"psd_snr_db"]=snr
                F.at[idx,"psd_peak_hz"]=np.nan
                F.at[idx,"psd_peak_power"]=np.nan

        print(f"✓ subject {int(subj):02d} done (HILBERT/PRV/FUSED) from {sig_csv.name}")

    # 저장
    out_h = f"{args.out_base}_hilbert.parquet"
    out_p = f"{args.out_base}_prv.parquet"
    out_f = f"{args.out_base}_fused.parquet"
    Path(Path(out_h).parent).mkdir(parents=True, exist_ok=True)
    H.to_parquet(out_h, index=False)
    P.to_parquet(out_p, index=False)
    F.to_parquet(out_f, index=False)
    print(f"Saved:\n  {out_h}\n  {out_p}\n  {out_f}")

if __name__ == "__main__":
    main()

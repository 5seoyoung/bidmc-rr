# src/make_windows.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# 설정
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "data" / "bidmc_dataset" / "bidmc-ppg-and-respiration-dataset-1.0.0" / "bidmc_csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "windows.parquet"

# 윈도우 파라미터 (필요시 조정)
WINDOW_SEC = 30.0
STEP_SEC = 10.0

# 호흡 대역(Hz) — 일반 성인 호흡 대역
RESP_BAND: Tuple[float, float] = (0.10, 0.50)


# ---------------------------
# 유틸
# ---------------------------
def _normalize_col(col: str) -> str:
    """컬럼명 공백/대소문자/특수문자 통일"""
    c = re.sub(r"\s+", " ", col.strip())
    c = c.lower()
    c = c.replace("_", " ")
    return c


def _choose_first(series: Iterable[Optional[float]]) -> Optional[float]:
    for x in series:
        if x is not None and not (isinstance(x, float) and np.isnan(x)):
            return x
    return None


def _infer_fs_from_time(t: np.ndarray) -> Optional[float]:
    if t is None or len(t) < 3:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return None
    return float(1.0 / np.median(dt))


def _parse_fix_for_fs(fix_path: Path) -> Optional[float]:
    """Fix.txt에서 샘플링 주파수 추정 (없으면 None)"""
    if not fix_path.exists():
        return None
    txt = fix_path.read_text(errors="ignore")
    # 예: "signal sample frequency: 125 Hz", "sampling rate = 62.5 Hz" 등
    m = re.search(r"(?i)(sample|sampling).{0,15}(freq|rate).{0,10}([\d.]+)\s*hz", txt)
    if m:
        try:
            return float(m.group(3))
        except Exception:
            return None
    return None


def _find_ppg_column(df_sig: pd.DataFrame) -> Optional[str]:
    """PPG/PLETH 컬럼 자동 탐색"""
    candidates = []
    for c in df_sig.columns:
        cn = _normalize_col(c)
        if any(k in cn for k in ["ppg", "pleth", "pulse", "ir pleth", "ir-pleth", "red pleth"]):
            candidates.append(c)
    if candidates:
        return candidates[0]
    # 그래도 못 찾으면 숫자형 컬럼 중 첫 번째 (차선책)
    numeric_cols = [c for c in df_sig.columns if pd.api.types.is_numeric_dtype(df_sig[c])]
    return numeric_cols[0] if numeric_cols else None


def _read_signals(signals_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], List[str]]:
    """Signals.csv 읽고 (time, ppg, fs) 반환. 경고 메시지 리스트 포함"""
    warns: List[str] = []
    if not signals_path.exists():
        warns.append(f"Signals.csv 없음: {signals_path.name}")
        return None, None, None, warns

    df = pd.read_csv(signals_path)
    # 시간축 후보
    time_cols = [c for c in df.columns if "time" in _normalize_col(c)]
    t = None
    if time_cols:
        t = df[time_cols[0]].to_numpy(dtype=float)
    # PPG 컬럼 찾기
    ppg_col = _find_ppg_column(df)
    if ppg_col is None:
        warns.append(f"PPG/PLETH 컬럼을 찾지 못함: {signals_path.name}")
        return t, None, None, warns
    x = df[ppg_col].to_numpy(dtype=float)

    # fs 추정
    fs = _infer_fs_from_time(t) if t is not None else None
    return t, x, fs, warns


def _extract_breath_samples_from_df(df_breaths: pd.DataFrame, verbose_cols: bool = True) -> np.ndarray:
    """
    Breaths.csv에서 호흡 어노테이션의 'signal sample no'를 robust 하게 추출.
    - 공백/대소문자/복수형/밑줄 변형 대응
    - 두 개 ann 컬럼이 있을 수 있음 (ann1, ann2)
    """
    if df_breaths is None or df_breaths.empty:
        return np.array([], dtype=int)

    norm_map = {c: _normalize_col(c) for c in df_breaths.columns}

    def _is_sample_col(nc: str) -> bool:
        # 예: "breaths ann1 [signal sample no]" / " breaths ann2 [signal sample no]" / "breath ann [signal sample number]"
        return ("breath" in nc and "ann" in nc and "signal" in nc and "sample" in nc)

    candidates = [orig for orig, nc in norm_map.items() if _is_sample_col(nc)]
    if not candidates:
        # 좀 더 완화: "ann"이 없더라도 "breath" + "signal sample" 조합 찾기
        candidates = [
            orig for orig, nc in norm_map.items()
            if ("breath" in nc and "signal" in nc and "sample" in nc)
        ]

    if not candidates:
        if verbose_cols:
            print("Breaths.csv 컬럼 확인 필요:", list(df_breaths.columns))
        return np.array([], dtype=int)

    vals: List[int] = []
    for c in candidates:
        s = pd.to_numeric(df_breaths[c], errors="coerce").dropna().astype(int).tolist()
        vals.extend(s)
    if not vals:
        return np.array([], dtype=int)
    vals = sorted(set(v for v in vals if v >= 0))
    return np.asarray(vals, dtype=int)


def _read_breaths(breaths_path: Path) -> np.ndarray:
    if not breaths_path.exists():
        return np.array([], dtype=int)
    df = pd.read_csv(breaths_path)
    return _extract_breath_samples_from_df(df, verbose_cols=True)


# ---------------------------
# 신호 품질 지표: 대역 전력 비(band power ratio)
# ---------------------------
def _periodogram(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    SciPy 없이 Numpy로 간단 periodogram (one-sided)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0 or not np.isfinite(x).any():
        return np.array([]), np.array([])
    x = x - np.nanmean(x)
    # NaN 처리
    if np.isnan(x).any():
        x = np.nan_to_num(x, nan=0.0)
    # FFT
    X = np.fft.rfft(x, n=n)
    Pxx = (np.abs(X) ** 2) / (fs * n)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    return Pxx, f


def band_power_ratio(signal_or_Pxx, fs_or_f, band: Tuple[float, float]) -> float:
    """
    두 가지 호출 모두 지원:
      1) band_power_ratio(x, fs, band)          # x: 시계열
      2) band_power_ratio(Pxx, f, band)         # PSD와 주파수 벡터

    반환: (band 파워) / (전체 파워), 실패시 np.nan
    """
    # 입력 분기
    if np.isscalar(fs_or_f):
        # 시계열 + fs
        x = np.asarray(signal_or_Pxx, dtype=float)
        fs = float(fs_or_f)
        if x is None or len(x) == 0 or not np.isfinite(fs) or fs <= 0:
            return np.nan
        Pxx, f = _periodogram(x, fs)
    else:
        # 이미 PSD + f
        Pxx = np.asarray(signal_or_Pxx, dtype=float)
        f = np.asarray(fs_or_f, dtype=float)

    # 방어적 체크
    if Pxx is None or f is None or len(Pxx) == 0 or len(f) == 0:
        return np.nan
    if np.isnan(Pxx).all() or np.isnan(f).all():
        return np.nan

    f_low, f_high = band
    m_all = np.isfinite(Pxx) & np.isfinite(f) & (f >= 0)
    m_band = m_all & (f >= f_low) & (f <= f_high)

    if not np.any(m_all) or not np.any(m_band):
        return np.nan

    # Deprecation 대응: np.trapz -> np.trapezoid
    tot = float(np.trapezoid(Pxx[m_all], f[m_all])) if np.any(m_all) else np.nan
    bp = float(np.trapezoid(Pxx[m_band], f[m_band])) if np.any(m_band) else np.nan

    if not np.isfinite(tot) or tot <= 0:
        return np.nan
    if not np.isfinite(bp) or bp < 0:
        return np.nan

    return bp / tot


# ---------------------------
# 윈도우 생성
# ---------------------------
@dataclass
class SubjectPaths:
    signals: Path
    numerics: Path
    breaths: Path
    fix: Path


def _paths_for_subject(csv_dir: Path, sid: int) -> SubjectPaths:
    base = f"bidmc_{sid:02d}_"
    return SubjectPaths(
        signals=csv_dir / f"{base}Signals.csv",
        numerics=csv_dir / f"{base}Numerics.csv",
        breaths=csv_dir / f"{base}Breaths.csv",
        fix=csv_dir / f"{base}Fix.txt",
    )


def _make_windows_for_subject(sid: int, csv_dir: Path) -> pd.DataFrame:
    p = _paths_for_subject(csv_dir, sid)

    # 신호 읽기
    t, x, fs, warns = _read_signals(p.signals)
    for w in warns:
        print(w)

    # fs 추론 실패 시 Fix.txt 시도
    if (fs is None or not np.isfinite(fs) or fs <= 0) and p.fix.exists():
        fs = _parse_fix_for_fs(p.fix)

    if fs is None or not np.isfinite(fs) or fs <= 0:
        # 시간 컬럼도 없고 fs도 모르면 스킵
        print(f"샘플링 주파수 추정 실패 → subject {sid:02d} 스킵")
        return pd.DataFrame(columns=["subject", "t_start", "t_end", "sqi", "rr"])

    # 시간축 없으면 생성
    if t is None or len(t) != len(x):
        t = np.arange(len(x), dtype=float) / fs

    # Breaths 읽기 (signal sample index 기준)
    breath_samples = _read_breaths(p.breaths)

    # 윈도우 인덱스
    win_len = int(round(WINDOW_SEC * fs))
    step_len = int(round(STEP_SEC * fs))
    if win_len <= 1 or step_len <= 0:
        return pd.DataFrame(columns=["subject", "t_start", "t_end", "sqi", "rr"])

    rows = []
    start = 0
    n = len(x)
    while start + win_len <= n:
        stop = start + win_len
        seg = x[start:stop]

        # SQI: 호흡 대역 전력비
        sqi = band_power_ratio(seg, fs, band=RESP_BAND)

        # RR 라벨(대략): 윈도우 구간에 포함된 breath 어노테이션 개수 → 분당 환산
        if breath_samples.size > 0:
            m = (breath_samples >= start) & (breath_samples < stop)
            breaths_count = int(np.count_nonzero(m))
            rr = breaths_count * (60.0 / WINDOW_SEC)
        else:
            rr = np.nan

        rows.append(
            {
                "subject": sid,
                "t_start": float(t[start]),
                "t_end": float(t[stop - 1]),
                "sqi": float(sqi) if np.isfinite(sqi) else np.nan,
                "rr": float(rr) if np.isfinite(rr) else np.nan,
            }
        )
        start += step_len

    return pd.DataFrame(rows)


# ---------------------------
# 메인
# ---------------------------
def main():
    csv_dir = CSV_DIR
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV_DIR가 존재하지 않습니다: {csv_dir}")

    all_rows = []
    # BIDMC 배포 기준 53명
    for sid in range(1, 54):
        df_sub = _make_windows_for_subject(sid, csv_dir)
        if not df_sub.empty:
            all_rows.append(df_sub)
            print(f"✓ subject {sid:02d} done")
        else:
            print(f"subject {sid:02d} 결과 없음(건너뜀)")

    if not all_rows:
        print("생성된 윈도우가 없습니다. 입력 데이터/경로를 확인하세요.")
        return

    df = pd.concat(all_rows, ignore_index=True)

    # pyarrow 또는 fastparquet 필요 (pyarrow 설치 완료라고 하셨으니 그대로 사용)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

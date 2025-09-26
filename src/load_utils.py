# src/load_utils.py
import os, glob, numpy as np, pandas as pd
import re

PPG_KEYS  = ["ppg", "pleth", "plethysm", "plethysmo"]
ECG_KEYS  = ["ecg", "ekg", "lead", "ii"]
RESP_KEYS = ["resp", "imp", "impedance"]
TIME_KEYS = ["time", "t"]

def find_col(cols, keys):
    low = {c.lower(): c for c in cols}
    for c_low, c in low.items():
        for k in keys:
            if k in c_low:
                return c
    return None

def infer_fs_from_time(df, time_col, default_fs=125.0):
    if time_col is None or time_col not in df.columns: 
        return default_fs
    t = df[time_col].to_numpy()
    if len(t) < 3: return default_fs
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt>0)]
    if len(dt)==0: return default_fs
    fs = 1.0/np.median(dt)
    return 125.0 if 110 <= fs <= 140 else float(fs)

def _find_csv_dir(base):
    root = os.path.join(base, "data", "bidmc_dataset")
    candidates = glob.glob(os.path.join(root, "**", "bidmc_csv"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"'bidmc_csv' 폴더를 {root} 하위에서 찾을 수 없습니다.")
    return candidates[0]

def get_paths(base):
    CSV_DIR = _find_csv_dir(base)
    sig_files = sorted(glob.glob(os.path.join(CSV_DIR, "bidmc_*_Signals.csv")))
    num_files = sorted(glob.glob(os.path.join(CSV_DIR, "bidmc_*_Numerics.csv")))
    brt_files = sorted(glob.glob(os.path.join(CSV_DIR, "bidmc_*_Breaths.csv")))
    fix_files = sorted(glob.glob(os.path.join(CSV_DIR, "bidmc_*_Fix.txt")))
    return CSV_DIR, sig_files, num_files, brt_files, fix_files

def load_subject(base, idx=1):
    CSV_DIR, sig_files, num_files, brt_files, fix_files = get_paths(base)
    sig = pd.read_csv(sig_files[idx-1])
    num = pd.read_csv(num_files[idx-1])
    brt = pd.read_csv(brt_files[idx-1])
    with open(fix_files[idx-1], "r", encoding="utf-8", errors="ignore") as f:
        fix_txt = f.read()

    time_col = find_col(sig.columns, TIME_KEYS)
    fs = infer_fs_from_time(sig, time_col)

    ppg_col  = find_col(sig.columns, PPG_KEYS)
    ecg_col  = find_col(sig.columns, ECG_KEYS)
    resp_col = find_col(sig.columns, RESP_KEYS)

    brt_cols = {c.lower(): c for c in brt.columns}
    breath_samples, breath_times = None, None
    if "sample" in brt_cols:
        breath_samples = brt[brt_cols["sample"]].to_numpy(dtype=float)
        breath_times = breath_samples / fs
    elif "time" in brt_cols:
        breath_times = brt[brt_cols["time"]].to_numpy(dtype=float)
        breath_samples = breath_times * fs
    else:
        print("Breaths.csv 컬럼 확인 필요:", brt.columns.tolist())

    return {
        "fs": fs, "time_col": time_col,
        "ppg_col": ppg_col, "ecg_col": ecg_col, "resp_col": resp_col,
        "signals": sig, "numerics": num, "breaths": brt,
        "breath_times": breath_times, "breath_samples": breath_samples,
        "fix_text": fix_txt
    }

def extract_breath_samples_from_df(breaths_df: pd.DataFrame, fs: float):
    """
    Breaths.csv의 컬럼명이 파일마다 미묘하게 다른 문제를 처리.
    - 컬럼명 앞뒤 공백 제거
    - 'breath'와 'sample'을 포함하는 컬럼 자동 탐색 (ann1/ann2 등 다양성 허용)
    - 숫자만 모아 중복 제거 후 정렬
    - sample index -> seconds 로 변환

    Parameters
    ----------
    breaths_df : pd.DataFrame  # Breaths.csv 로드 결과
    fs         : float         # signal sampling rate (Hz)

    Returns
    -------
    samples : np.ndarray[int]    # 호흡 이벤트 샘플 인덱스 (정수)
    times   : np.ndarray[float]  # 초 단위 타임스탬프 (samples/fs)
    """
    if breaths_df is None or len(breaths_df) == 0 or not np.isfinite(fs) or fs <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    br = breaths_df.copy()
    # 1) 컬럼명 앞뒤 공백 제거
    br.columns = [str(c).strip() for c in br.columns]

    # 2) 후보 컬럼 자동 탐색
    cand = [c for c in br.columns if ("breath" in c.lower()) and ("sample" in c.lower())]

    # 3) 흔한 대안 컬럼명도 허용 (필요시 확장)
    if len(cand) == 0:
        alt_candidates = []
        lower_cols = {c.lower(): c for c in br.columns}
        for key in ["ann1", "ann2", "breaths ann1 [signal sample no]", "breaths ann2 [signal sample no]"]:
            if key in lower_cols:
                alt_candidates.append(lower_cols[key])
        cand = alt_candidates

    # 그래도 못 찾았으면 원본 컬럼 리스트를 출력하고 빈 결과 리턴
    if len(cand) == 0:
        print(f"Breaths.csv 컬럼 확인 필요: {list(breaths_df.columns)}")
        return np.array([], dtype=int), np.array([], dtype=float)

    # 4) 후보 컬럼들을 세로로 쌓아 숫자만 수집 → 정수 변환 → 고유/정렬
    stacked = pd.to_numeric(br[cand].stack(), errors="coerce").dropna()
    if stacked.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    samples = np.unique(stacked.astype(int).to_numpy())
    times = samples.astype(float) / float(fs)
    return samples, times


def extract_breath_samples_from_df(breaths_df: pd.DataFrame, fs: float):
    """
    Breaths.csv 컬럼명이 파일마다 '공백/복수형/ann1, ann2' 등으로 제각각인 문제를 견고하게 처리.
    - 컬럼명: 연속 공백 1개로 축소 + strip
    - 'breath' & 'sample' 조합 또는 'ann\\d' & 'sample' 조합 탐색
    - 숫자만 모아 고유/정렬 → sample index, seconds 반환
    """
    if breaths_df is None or len(breaths_df) == 0 or not np.isfinite(fs) or fs <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    br = breaths_df.copy()
    br.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in br.columns]

    cands = []
    for c in br.columns:
        lc = c.lower()
        if ('breath' in lc and 'sample' in lc) or (re.search(r'ann\s*\d', lc) and 'sample' in lc):
            cands.append(c)

    if not cands:
        # 여분 fallback: ann1/ann2가 들어가면 사용
        for c in br.columns:
            lc = c.lower()
            if ('ann1' in lc or 'ann2' in lc) and 'sample' in lc:
                cands.append(c)

    if not cands:
        # 끝까지 못 찾으면 컬럼명 알려주고 빈 결과
        print(f"Breaths.csv 컬럼 확인 필요: {list(breaths_df.columns)}")
        return np.array([], dtype=int), np.array([], dtype=float)

    stacked = pd.to_numeric(br[cands].stack(), errors="coerce").dropna()
    if stacked.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    samples = np.unique(stacked.astype(np.int64).to_numpy())
    times   = samples.astype(float) / float(fs)
    return samples, times

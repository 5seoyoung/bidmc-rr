# src/setup_bidmc.py
import os, glob, sys

BASE = os.path.dirname(os.path.dirname(__file__))
ROOT = os.path.join(BASE, "data", "bidmc_dataset")

# bidmc_csv 폴더 자동 탐색
candidates = glob.glob(os.path.join(ROOT, "**", "bidmc_csv"), recursive=True)
if not candidates:
    print(f" 'bidmc_csv' 폴더를 찾지 못했습니다. 경로를 확인하세요:\n  {ROOT}")
    sys.exit(1)

CSV_DIR = candidates[0]
print(f"CSV_DIR: {CSV_DIR}")

def cnt(pat): 
    return len(glob.glob(os.path.join(CSV_DIR, pat)))

print("샘플 5개:", sorted(os.listdir(CSV_DIR))[:5])
print(f"Signals: {cnt('bidmc_*_Signals.csv')}, "
      f"Numerics: {cnt('bidmc_*_Numerics.csv')}, "
      f"Breaths: {cnt('bidmc_*_Breaths.csv')}, "
      f"Fix: {cnt('bidmc_*_Fix.txt')}")

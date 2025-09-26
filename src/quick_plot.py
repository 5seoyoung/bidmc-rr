import os, numpy as np, matplotlib.pyplot as plt
from load_utils import load_subject

BASE = os.path.dirname(os.path.dirname(__file__))
D = load_subject(BASE, idx=1)

print("FS:", D["fs"])
print("Time:", D["time_col"], "/ PPG:", D["ppg_col"], "/ ECG:", D["ecg_col"], "/ RESP:", D["resp_col"])
print("\nFix.txt (앞부분):\n", "\n".join(D["fix_text"].splitlines()[:6]))

sig = D["signals"]
time_col = D["time_col"] or "Time"
if time_col not in sig.columns:
    sig[time_col] = np.arange(len(sig)) / D["fs"]

t = sig[time_col].to_numpy()
t0, t1 = t[0], t[0] + 10.0
m = (t>=t0) & (t<=t1)

def plot_one(colname, title):
    if colname and colname in sig.columns:
        plt.figure(figsize=(12,3))
        plt.plot(t[m], sig.loc[m, colname].to_numpy())
        plt.title(title)
        plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
        plt.tight_layout(); plt.show()

plot_one(D["ppg_col"],  f"PPG (first 10s) — {D['ppg_col']}")
plot_one(D["ecg_col"],  f"ECG (first 10s) — {D['ecg_col']}")
plot_one(D["resp_col"], f"Resp (first 10s) — {D['resp_col']}")

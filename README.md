# PPG-RR-Trust

**Lightweight PPG Respiratory Rate (RR) pipeline with SQI·SNR gating and conformal prediction intervals**
Oh Seoyoung (Kookmin University)

---

## 1. Description

This repository implements a lightweight, reproducible pipeline that estimates respiratory rate (RR) from PPG signals on the public BIDMC dataset and outputs **prediction intervals** alongside point estimates. Three estimators—frequency (PSD), envelope (Hilbert), and pulse rate variability (PRV)—are fused via **softmax weighting**, and **SQI·SNR gating** plus **Conformal Prediction** quantify “when the estimate is trustworthy.”

Why it matters
In clinical and wearable settings, RR is a vital sign. Beyond raw accuracy, one must present **output trustworthiness** and **usable coverage** to enable safe real-world use. This pipeline is computationally light and interpretable, making it portable and reproducible.

Our contributions

* **Lightweight & interpretable**: combine PSD, envelope, and PRV with simple fusion for high efficiency/portability
* **Trust included**: conformal intervals achieve target coverage and make uncertainty explicit
* **Operational policy**: SQI·SNR gating quantifies the retention–accuracy–coverage trade-off

---

## 2. Dataset

* **BIDMC PPG and Respiration Dataset v1.0.0** (PhysioNet)
  Overview/Download: [https://physionet.org/content/bidmc/1.0.0/](https://physionet.org/content/bidmc/1.0.0/)
  
The dataset is not bundled due to license/size. Please download from the link and place as:

```
data/
└─ bidmc_dataset/
   └─ bidmc-ppg-and-respiration-dataset-1.0.0/
      └─ bidmc_csv/   # bidmc_XX_Signals.csv per subject
```

---

## 3. Quickstart (minimal steps)

```bash
# 1) (optional) venv & deps
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy matplotlib pyarrow fastparquet scikit-learn jinja2

# 2) Generate per-window PPG outputs
python src/add_psd_baseline_to_windows.py \
  --csv-root "data/bidmc_dataset/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc_csv" \
  --out data/processed/windows_with_pred.parquet

# 3) Fusion sweep (softmax τ, PRV weight cap) → best fusion
python scripts/sweep_fusion.py \
  --paths  data/processed/windows_with_pred.parquet \
          data/processed/windows_with_pred_hilbert.parquet \
          data/processed/windows_with_pred_prv.parquet \
  --labels raw hilbert prv \
  --rules argmax softmax --taus 0.2,0.4,0.6,0.8,1.0,2.0 \
  --cap-label prv --max-w 0.5 \
  --outdir data/derived \
  --best-out data/processed/windows_with_pred_fused_best.parquet

# 4) Two-track pipeline (All-comers / High-confidence) + Conformal
python scripts/two_track_pipeline.py \
  --path data/processed/windows_with_pred_fused_best.parquet \
  --outdir data/derived \
  --alpha-all 0.08 --alpha-gated 0.08 \
  --thr-sqi 0.048 --thr-snr -2.0 \
  --n-bins 3 --n-folds 5

# 5) Summary report (tables/figures)
python scripts/make_two_track_report.py \
  --all   data/processed/windows_with_pred_fused_best_all_a008.parquet \
  --gated data/processed/windows_with_pred_fused_best_gated_a008.parquet \
  --label-all all_a008 --label-gated gated_a008 \
  --outdir data/derived
```

---

## 4. Methods (brief)

* **Features**:
  (1) PPG PSD in 0.1–0.6 Hz (peak, SNR), (2) Hilbert-envelope-based respiratory rate, (3) PRV summaries
* **Fusion**: softmax weighting (temperature τ = 2.0), PRV weight capped at 0.5 to avoid dominance
* **Quality gating**: first-stage SQI threshold, second-stage SNR threshold; default **SQI ≥ 0.048, SNR ≥ −2.0 dB**
* **Uncertainty**: subject-wise 5-fold split conformal (α = 0.08) and SNR-stratified Mondrian (3 bins)

---

## 5. Results (key points)

* **All-comers**: MAE 7.25 bpm; conformal α = 0.08 → coverage 0.92, mean width ≈ 26.3 bpm
* **High-confidence** (gated; retention ≈ 64.8%): MAE 6.78 bpm; coverage 0.918; mean width ≈ 26.3 bpm
* **Operational tip**: with a 30 bpm width gate, both tracks maintain coverage ≥ 90%

---

## 6. Repository Structure

```
src/
  add_psd_baseline_to_windows.py
scripts/
  eval_rr.py
  eval_rr_2d_gate.py
  fuse_methods.py
  sweep_fusion.py
  add_conformal_pi.py
  add_conformal_pi_mondrian.py
  eval_conformal_pi.py
  two_track_pipeline.py
  make_two_track_report.py
data/
  processed/   # intermediate parquet artifacts
  derived/     # tables/figures (csv/png/tex)
```

---

## 7. References

* Charlton PH, et al. Physiol Meas. 2016. Assessment of RR algorithms from ECG/PPG.
* Pimentel MAF, et al. IEEE TBME. 2017. Toward robust RR estimation from pulse oximeters.
* Charlton PH, et al. IEEE Reviews in Biomedical Engineering. 2018. Review on RR from ECG/PPG.
* PhysioNet BIDMC Dataset: [https://physionet.org/content/bidmc/1.0.0/](https://physionet.org/content/bidmc/1.0.0/)

---

## 8. License & Contact

* Code: MIT (or adapt as needed)
* Data: follow PhysioNet dataset license
* Contact: Oh Seoyoung · Kookmin University · [5seo0_oh@kookmin.ac.kr](mailto:5seo0_oh@kookmin.ac.kr)

---

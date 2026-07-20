# Predictive Maintenance — Turbofan RUL Prediction & Bearing Fault Detection

A multi-stage industrial predictive maintenance project covering two
distinct sensor modalities, two distinct problem types, and full-stack
deployment with explainability — built to demonstrate breadth (not just
depth) across the ML engineering lifecycle.

| Stage | Sensor type | Task | Models | Status |
|---|---|---|---|---|
| 1 | Operational (temperature, pressure, fan speed) | RUL regression | XGBoost baseline | Done |
| 2 | Operational (same as above) | RUL regression | LSTM / GRU / Transformer | Done |
| 3 | Vibration / acoustic (accelerometer) | Fault-stage classification | Random Forest / CNN | Done |
| 4 | — | Deployment + explainability | Flask dashboards (both stages) | Done |

---

## Stage 1-2 — Remaining Useful Life Prediction

**Dataset:** NASA C-MAPSS — simulated turbofan engine sensor data (21
channels), run-to-failure, 4 difficulty variants (FD001-FD004).

**Approach:** classical baseline (XGBoost on hand-engineered rolling
features) compared against deep sequence models (LSTM, GRU, Transformer)
consuming raw sliding-window sensor sequences directly.

**Results (FD001, Test RMSE):**

| Model | Test RMSE |
|---|---|
| XGBoost (baseline) | 17.97 |
| LSTM | 14.72 |
| GRU | 14.43 |
| **Transformer** | **14.22** |

All three deep models outperformed the classical baseline by ~18-21%,
converging to similar performance after fixing an LSTM training-instability
issue (gradient clipping + a gentler learning rate). Validated across all
four C-MAPSS subsets — performance degrades on FD002/FD004 (6 operating
conditions vs. 1), consistent with known dataset difficulty and confirming
the pipeline behaves as domain knowledge predicts rather than by chance.

**Notebook:** `predictive_maintenance_colab.ipynb`

---

## Stage 3 — Vibration-Based Fault Detection

**Dataset:** NASA IMS Bearing Dataset — real high-frequency accelerometer
data from a physical run-to-failure test (Bearing 3, documented inner race
defect), a genuinely different data modality from Stages 1-2.

**Approach:** FFT/spectrogram signal processing, comparing hand-engineered
vibration features (RMS, kurtosis, frequency-band energies) feeding a
Random Forest against raw spectrograms feeding a CNN — classifying health
stage (Normal / Degrading / Near-Failure) rather than predicting a number.

**Results (leak-free, 3-class):**

| Model | Degrading (P/R) | Near-Failure (P/R) | Normal (P/R) |
|---|---|---|---|
| Random Forest | 0.00 / 0.00 | 0.43 / 0.75 | 1.00 / 0.77 |
| CNN (spectrograms) | 0.00 / 0.00 | 0.00 / 0.00 | 0.98 / 1.00* |

*CNN's high Normal accuracy reflects always predicting the majority class,
not genuine fault detection — stated plainly rather than hidden.

**Why this stage is the strongest methodology showcase in the project:**
building it surfaced four distinct real issues, each caught because a
result looked suspiciously good and got investigated rather than accepted:

1. A naive time-percentage labeling heuristic that didn't match the actual
   RMS trend — fixed with data-driven changepoint detection.
2. **Feature leakage** — RMS was used both to create labels and as a
   classifier input, letting the model trivially reconstruct its own
   ground truth. Fixed by restricting to independent spectral-shape
   features.
3. **Threshold-selection leakage** — a "best F1" search was tuning a
   decision threshold using the test set's own labels. Fixed by reporting
   the untuned default threshold instead.
4. **CNN training instability** from an extreme (~113x) class-weight ratio,
   causing the loss to bounce instead of converge — fixed with capped
   weights, verified via a direct loss-variance comparison.

**Conclusion:** with only 30 real anomalous examples total, hand-engineered
domain features (kurtosis specifically, consistent with known impulsive
bearing-fault signatures) meaningfully outperformed raw deep learning on
spectrograms — a legitimate, well-documented small-data ML finding, not a
shortcoming of the pipeline.

**Notebook:** `predictive_maintenance_stage3_vibration.ipynb`
**Detailed writeup:** `STAGE3_README.md`

---

## Stage 4 — Deployment & Explainability

Two Flask dashboards, one per stage, each honestly surfacing model
limitations rather than only showcasing best-case results.

### RUL Dashboard (`flask-dashboard/`)
- Live RUL prediction from the trained Transformer, for selectable demo
  engines
- Cockpit-style gauge with risk-level color coding (nominal / monitor /
  maintenance due)
- SHAP-based sensor importance panel (computed from the XGBoost baseline,
  since SHAP explains tree models cleanly while explaining a Transformer
  directly needs different techniques — disclosed explicitly in the UI)
- Runs on port 5000

### Vibration Fault Dashboard (`stage3-dashboard/`)
- Random Forest and CNN predictions shown side by side for demo bearing
  snapshots, sampled exclusively from the held-out test set (an earlier
  version accidentally included training examples in the demo pool — a
  demo-selection leak, caught and fixed the same way as the modeling
  leaks above)
- Spectrogram viewer and feature-importance chart
- A limitations banner directly in the UI, stating plainly that Degrading
  is undetectable and the CNN essentially always predicts Normal
- Runs on port 5001 (so both dashboards can run simultaneously)

---

## Repo Structure

```
predictive-maintenance/
├── README.md                                    <- this file
├── STAGE3_README.md                              <- detailed Stage 3 writeup
├── predictive_maintenance_colab.ipynb            <- Stages 1-2 (RUL)
├── predictive_maintenance_stage3_vibration.ipynb <- Stage 3 (fault detection)
├── flask-dashboard/                              <- Stage 4, RUL dashboard
│   ├── app.py
│   ├── templates/index.html
│   ├── requirements.txt
│   └── artifacts/          <- populated from the Stage 1-2 notebook's export
└── stage3-dashboard/                             <- Stage 4, vibration dashboard
    ├── app.py
    ├── templates/index.html
    ├── requirements.txt
    └── artifacts/          <- populated from the Stage 3 notebook's export
```

## How to run everything

1. **Stages 1-2**: open `predictive_maintenance_colab.ipynb` in Google
   Colab, enable a GPU, run top to bottom through the export section.
   Download `artifacts.zip`, extract into `flask-dashboard/artifacts/`,
   then `pip install -r requirements.txt && python app.py` (port 5000).

2. **Stage 3**: open `predictive_maintenance_stage3_vibration.ipynb` in
   Colab, enable a GPU, run top to bottom (Section 1b offers Google Drive
   caching so you don't have to re-download/re-process the ~1GB dataset on
   every session). Download `stage3_artifacts.zip`, extract into
   `stage3-dashboard/artifacts/`, then `pip install -r requirements.txt &&
   python app.py` (port 5001).

3. Both dashboards can run at the same time, on their respective ports.

## What this project demonstrates

- **Breadth across sensor modalities and problem types**: tabular
  time-series regression (Stages 1-2) and raw signal-processing
  classification (Stage 3), rather than three variations of the same
  approach.
- **Multi-architecture comparison with honest reporting**: baseline before
  deep learning, imperfect results stated plainly rather than optimized
  away, and explicit discussion of why simpler methods sometimes win
  (Stage 3's domain-features-beat-CNN finding).
- **Rigorous methodology, not just model-building**: four distinct forms
  of leakage or instability were found and fixed during development, each
  through the same discipline — treating a suspiciously good result as a
  signal to investigate, not a result to report.
- **Full-stack delivery**: both stages ship as working, explainable, and
  honestly-limited dashboards, not just notebooks with a final metric.

# Predictive Maintenance Dashboard

Flask app that serves live RUL predictions using the best deep model trained
in the Colab notebook (LSTM/GRU/Transformer, whichever won), plus a SHAP-based
explainability panel from the XGBoost baseline.

# Setup

1. **Run the Colab notebook through Stage 4** (Sections 16-18) — this trains
   everything and downloads `artifacts.zip`.

2. **Extract the artifacts** into this folder:
   ```
   flask-dashboard/
   ├── app.py
   ├── templates/index.html
   ├── requirements.txt
   └── artifacts/          <-- extract artifacts.zip contents here
       ├── best_model.pt
       ├── model_config.json
       ├── xgb_baseline.json
       ├── scaler.pkl
       ├── feature_cols.json
       ├── shap_values.npy
       ├── X_test_baseline.csv
       ├── demo_sequences.npy
       └── demo_true_rul.npy
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run:**
   ```bash
   python app.py
   ```

5. Open **http://127.0.0.1:5000** in your browser.

# What it shows

- A cockpit-style gauge showing predicted Remaining Useful Life (RUL) for a
  selectable demo engine, color-coded by risk (green = nominal, amber =
  monitor, red = maintenance due)
- Predicted vs. true RUL comparison for each demo engine
- A SHAP feature-importance chart showing which sensors most influence RUL
  predictions, computed from the XGBoost baseline

# Design note on explainability

The dashboard's live predictions come from the **deep model** (best of
LSTM/GRU/Transformer), but the SHAP explainability panel is computed from the
**XGBoost baseline**. This is intentional and disclosed in the UI: SHAP's
`TreeExplainer` gives fast, exact explanations for tree models, while
explaining a transformer/RNN directly requires different (slower, noisier)
techniques like Integrated Gradients. Since both models are trained on the
same sensors, the baseline's feature-importance ranking is a reasonable proxy
for "which sensors matter" — worth stating clearly in interviews or write-ups
rather than implying the SHAP values explain the Transformer directly.

# Extending this

- Add a "Stage 3" panel for bearing vibration data (IMS dataset) if you build
  that module.
- Swap `demo_sequences.npy` for a file upload so users can test their own
  sensor windows.
- Add authentication if you ever deploy this somewhere public — right now
  it's a local demo app with no auth layer.

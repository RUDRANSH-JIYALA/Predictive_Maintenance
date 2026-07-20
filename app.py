"""
Vibration Fault Detection Dashboard — Flask app (Stage 3)

Serves fault-stage predictions (Normal / Degrading / Near-Failure) from both
the leak-free Random Forest and the CNN, for a set of demo bearing vibration
snapshots. Deliberately surfaces both models' known limitations rather than
implying more reliability than the underlying results support.

Setup:
    1. Extract stage3_artifacts.zip (downloaded from the Colab notebook)
       into the artifacts/ folder here.
    2. pip install -r requirements.txt
    3. python app.py
    4. Open http://127.0.0.1:5000
"""

import io
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, send_file

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

app = Flask(__name__)


# ---------------------------------------------------------------------------
# CNN architecture (must match the notebook exactly, so state_dict loads)
# ---------------------------------------------------------------------------

class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------

def load_artifacts():
    rf_model = joblib.load(os.path.join(ARTIFACT_DIR, "rf_model.pkl"))

    with open(os.path.join(ARTIFACT_DIR, "feature_cols.json")) as f:
        feature_cols = json.load(f)

    with open(os.path.join(ARTIFACT_DIR, "cnn_config.json")) as f:
        cnn_config = json.load(f)

    cnn_model = SpectrogramCNN(num_classes=len(cnn_config["classes"]))
    state_dict = torch.load(os.path.join(ARTIFACT_DIR, "cnn_model.pt"), map_location="cpu")
    cnn_model.load_state_dict(state_dict)
    cnn_model.eval()

    demo_df = pd.read_csv(os.path.join(ARTIFACT_DIR, "demo_snapshots.csv"))
    demo_spectrograms = np.load(os.path.join(ARTIFACT_DIR, "demo_spectrograms.npy"))

    with open(os.path.join(ARTIFACT_DIR, "limitations.json")) as f:
        limitations = json.load(f)

    return {
        "rf_model": rf_model,
        "feature_cols": feature_cols,
        "cnn_model": cnn_model,
        "cnn_classes": cnn_config["classes"],
        "demo_df": demo_df,
        "demo_spectrograms": demo_spectrograms,
        "limitations": limitations,
    }


ARTIFACTS = load_artifacts()

STATUS_COLORS = {"Normal": "low", "Degrading": "medium", "Near-Failure": "high"}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    demo_df = ARTIFACTS["demo_df"]
    return render_template(
        "index.html",
        num_demo=len(demo_df),
        demo_labels=demo_df["label"].tolist(),
        limitations=ARTIFACTS["limitations"],
    )


@app.route("/predict/<int:idx>")
def predict(idx):
    demo_df = ARTIFACTS["demo_df"]
    if idx < 0 or idx >= len(demo_df):
        return jsonify({"error": "index out of range"}), 400

    row = demo_df.iloc[idx]
    true_label = row["label"]

    # Random Forest prediction (pass a DataFrame with matching column names,
    # not a bare array, so scikit-learn doesn't warn about missing feature names)
    X = pd.DataFrame([row[ARTIFACTS["feature_cols"]].values], columns=ARTIFACTS["feature_cols"])
    rf_pred = ARTIFACTS["rf_model"].predict(X)[0]
    rf_proba = ARTIFACTS["rf_model"].predict_proba(X)[0]
    rf_classes = list(ARTIFACTS["rf_model"].classes_)
    rf_proba_dict = {cls: round(float(p), 3) for cls, p in zip(rf_classes, rf_proba)}

    # CNN prediction
    spec = ARTIFACTS["demo_spectrograms"][idx]
    with torch.no_grad():
        x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = ARTIFACTS["cnn_model"](x)
        cnn_proba = torch.softmax(logits, dim=1).numpy()[0]
    cnn_classes = ARTIFACTS["cnn_classes"]
    cnn_pred = cnn_classes[int(np.argmax(cnn_proba))]
    cnn_proba_dict = {cls: round(float(p), 3) for cls, p in zip(cnn_classes, cnn_proba)}

    return jsonify({
        "idx": idx,
        "true_label": true_label,
        "rf_prediction": rf_pred,
        "rf_proba": rf_proba_dict,
        "rf_status": STATUS_COLORS.get(rf_pred, "medium"),
        "cnn_prediction": cnn_pred,
        "cnn_proba": cnn_proba_dict,
        "cnn_status": STATUS_COLORS.get(cnn_pred, "medium"),
    })


@app.route("/spectrogram/<int:idx>.png")
def spectrogram_image(idx):
    demo_spectrograms = ARTIFACTS["demo_spectrograms"]
    if idx < 0 or idx >= len(demo_spectrograms):
        return jsonify({"error": "index out of range"}), 400

    spec = demo_spectrograms[idx]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/feature_importance.png")
def feature_importance_plot():
    rf_model = ARTIFACTS["rf_model"]
    feature_cols = ARTIFACTS["feature_cols"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()

    fig, ax = plt.subplots(figsize=(6, 4))
    importances.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, port=5001)

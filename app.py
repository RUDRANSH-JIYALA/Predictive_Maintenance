"""
Predictive Maintenance Dashboard — Flask app

Serves RUL predictions using the trained deep model (LSTM/GRU/Transformer,
whichever won in the Colab notebook) and shows global feature importance
from the XGBoost baseline via SHAP.

Setup:
    1. Extract artifacts.zip (downloaded from the Colab notebook, Stage 4)
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
matplotlib.use("Agg")  # no GUI backend needed for a server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, send_file

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Model architectures (must match the notebook exactly, so state_dict loads)
# ---------------------------------------------------------------------------

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, input_size, window_size, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        return self.fc(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------

def load_artifacts():
    with open(os.path.join(ARTIFACT_DIR, "model_config.json")) as f:
        config = json.load(f)

    architecture = config["architecture"]
    num_features = config["num_features"]
    window_size = config["window_size"]

    if architecture == "LSTM":
        model = LSTMRegressor(num_features)
    elif architecture == "GRU":
        model = GRURegressor(num_features)
    elif architecture == "Transformer":
        model = TransformerRegressor(num_features, window_size)
    else:
        raise ValueError(f"Unknown architecture in model_config.json: {architecture}")

    state_dict = torch.load(
        os.path.join(ARTIFACT_DIR, "best_model.pt"), map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()

    demo_sequences = np.load(os.path.join(ARTIFACT_DIR, "demo_sequences.npy"))
    demo_true_rul = np.load(os.path.join(ARTIFACT_DIR, "demo_true_rul.npy"))

    shap_values = np.load(os.path.join(ARTIFACT_DIR, "shap_values.npy"))
    X_test_baseline = pd.read_csv(os.path.join(ARTIFACT_DIR, "X_test_baseline.csv"))

    return {
        "model": model,
        "config": config,
        "demo_sequences": demo_sequences,
        "demo_true_rul": demo_true_rul,
        "shap_values": shap_values,
        "feature_names": list(X_test_baseline.columns),
    }


ARTIFACTS = load_artifacts()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def predict_rul(sequence: np.ndarray) -> float:
    """sequence shape: (window_size, num_features)"""
    with torch.no_grad():
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # add batch dim
        pred = ARTIFACTS["model"](x).item()
    return max(0.0, pred)  # RUL can't be negative


def risk_level(rul: float) -> str:
    if rul < 20:
        return "high"
    elif rul < 50:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    num_demo = len(ARTIFACTS["demo_sequences"])
    architecture = ARTIFACTS["config"]["architecture"]
    test_rmse = ARTIFACTS["config"]["test_rmse"]
    return render_template(
        "index.html",
        num_demo=num_demo,
        architecture=architecture,
        test_rmse=round(test_rmse, 2),
    )


@app.route("/predict/<int:engine_idx>")
def predict(engine_idx):
    demo_sequences = ARTIFACTS["demo_sequences"]
    demo_true_rul = ARTIFACTS["demo_true_rul"]

    if engine_idx < 0 or engine_idx >= len(demo_sequences):
        return jsonify({"error": "engine index out of range"}), 400

    sequence = demo_sequences[engine_idx]
    predicted_rul = predict_rul(sequence)
    true_rul = float(demo_true_rul[engine_idx])

    return jsonify({
        "engine_idx": engine_idx,
        "predicted_rul": round(predicted_rul, 1),
        "true_rul": round(true_rul, 1),
        "risk": risk_level(predicted_rul),
        "architecture": ARTIFACTS["config"]["architecture"],
    })


@app.route("/shap_plot.png")
def shap_plot():
    """Global feature importance bar chart from the XGBoost baseline's SHAP
    values. This explains the baseline model's sensor-importance ranking —
    a reasonable proxy for which sensors matter, even though the deep model
    (not XGBoost) is what's making live predictions above."""
    shap_values = ARTIFACTS["shap_values"]
    feature_names = ARTIFACTS["feature_names"]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[-10:]  # top 10

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        [feature_names[i] for i in order],
        mean_abs_shap[order],
        color="#4C72B0",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top Sensor Contributions (XGBoost Baseline)")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, port=5000)

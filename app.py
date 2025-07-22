#load libraries
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import requests

# ====== Configuration ======
MODEL_URL = "https://drive.google.com/file/d/1N5i1pXashLkOd-9p5WbEAE5dmptbdxyw/view?usp=sharing"  # Replace with your actual file ID
MODEL_PATH = "gru_seizure_prediction_model.keras"
SCALER_PATH = "scaler.pkl"
PCA_PATH = "pca.pkl"

# ====== Model Download ======
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Model download complete.")

download_model()

# ====== Load model and preprocessing tools ======
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
# pca = joblib.load(PCA_PATH)  # Uncomment if using PCA

# ====== Flask App ======
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)

        # Preprocessing
        features = scaler.transform(features)
        # features = pca.transform(features)  # Uncomment if using PCA
        features = features.reshape(features.shape[0], features.shape[1], 1)

        # Prediction
        prediction = model.predict(features)
        pred_label = int(prediction[0][0] > 0.5)

        return jsonify({
            "prediction": pred_label,
            "confidence": float(prediction[0][0])
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

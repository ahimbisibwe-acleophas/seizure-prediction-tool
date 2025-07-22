from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Load model and pre-processing tools
model = load_model("gru_seizure_prediction_model.keras")
scaler = joblib.load("scaler.joblib")  # If you saved your scaler
pca = joblib.load("pca.joblib")        # Optional PCA

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)

        # Preprocessing
        features = scaler.transform(features)
        # features = pca.transform(features)  # Uncomment if PCA was used
        features = features.reshape(features.shape[0], features.shape[1], 1)

        prediction = model.predict(features)
        pred_label = int(prediction[0][0] > 0.5)

        return jsonify({"prediction": pred_label, "confidence": float(prediction[0][0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# app.py
import os 
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify


# ___Config ___
MODEL_PATH = os.gentenv("MODEL_PATH", "model/iris_model.pkl") # adjust filname if needed 

# _____App_____
app = Flask(__name__)

# Load once at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
   
    # Fail fast with helpful massege 
    raise RuntimeError(f"Could not load from {MODEL_PATH}: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}, 200

@app.post("/predict")
def predict():
    """

    Accepts either:
    {"input": [[...feature vector...], [...]]} # 2d list
    or 
    {"input": [...feature vector...]}  # 1d list 
 
    try:
        payload = request.get_json(force=True)
        x = played.get("input")
        if x is None:
            return jsonify(error="Missing 'input'"), 400

        # Normalize to 2D array
        if isinstance(x, list) and (len)(x) > 0) and  not isinstance(x[0], list):
            x = [X]
       
        X = np.array(X, dtype= float)
        preds = model.predict(X)
        # If your model returns numpy types, convert to python
        preds = preds.tolist()
        return jsonify(prediction=preds), 200

   except Exceptoon as e:
       return jsonify(error=str(e)), 500

if __name__ == "__main__":
    # Local dev only; Render will run with Gunicorn (see startcommand below)
    app.run(host="0.0.0.0", port=int(os.eviron.get("PORT", 8000)))
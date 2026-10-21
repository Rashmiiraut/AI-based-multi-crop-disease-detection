import json
import os
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "crop_disease_model.keras")
CLASS_PATH = os.path.join(BASE, "model", "class_names.json")
METRICS_PATH = os.path.join(BASE, "model", "metrics.json")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ALLOWED = {"jpg", "jpeg", "png"}
COTTON = {"diseased cotton leaf", "diseased cotton plant",
          "fresh cotton leaf", "fresh cotton plant"}
CORN = {"blight",  "healthy"}

def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def crop_type(name):
    n = name.lower().strip()
    if n in COTTON or "cotton" in n:
        return "Cotton"
    if n in CORN or n in {"corn", "maize"}:
        return "Corn / Maize"
    return "Unknown"

@app.route("/")
def index():
    metrics = {}
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass
    return render_template("index.html", metrics=metrics)

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_available": os.path.exists(MODEL_PATH),
        "classes_available": os.path.exists(CLASS_PATH)
    })

@app.route("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"available": False})
    with open(METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    data["available"] = True
    return jsonify(data)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "Please select an image."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"success": False, "message": "Please select an image."}), 400
    if not allowed(file.filename):
        return jsonify({"success": False, "message": "Only JPG, JPEG and PNG images are supported."}), 400

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_PATH):
        return jsonify({
            "success": False,
            "message": "Model not trained yet. Run 'python train.py' first."
        }), 503

    try:
        image = Image.open(file.stream).convert("RGB")
        with open(CLASS_PATH, encoding="utf-8") as f:
            classes = json.load(f)

        model = load_model(MODEL_PATH)
        size = tuple(model.input_shape[1:3])
        image = image.resize(size)

        x = np.asarray(image, dtype=np.float32)
        x = np.expand_dims(x, 0)
        x = preprocess_input(x)

        probs = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        predicted = str(classes[idx])
        confidence = float(probs[idx] * 100)

        return jsonify({
            "success": True,
            "crop": crop_type(predicted),
            "prediction": predicted,
            "confidence": round(confidence, 2),
            "probabilities": {
                str(classes[i]): round(float(probs[i] * 100), 2)
                for i in range(len(classes))
            }
        })
    except UnidentifiedImageError:
        return jsonify({"success": False, "message": "Invalid image file."}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"success": False, "message": "Prediction failed. Check the trained model."}), 500

@app.errorhandler(413)
def too_large(_):
    return jsonify({"success": False, "message": "Image is too large. Maximum is 8 MB."}), 413

if __name__ == "__main__":
    app.run(debug=True)

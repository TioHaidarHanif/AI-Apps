from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
CORS(app)  # biar bisa dipanggil dari browser

# Load pretrained model
model = MobileNetV2(weights="imagenet")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # Konversi gambar ke RGB & resize
        img = Image.open(io.BytesIO(file.read()))
        
        # Convert to RGB if image has 4 channels (RGBA) or other formats
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Prediksi
        preds = model.predict(img_array)
        results = decode_predictions(preds, top=3)[0]

        predictions = [
            {"label": label, "description": desc, "probability": float(prob)}
            for (label, desc, prob) in results
        ]

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

import os, io
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
import sys, json
import boto3

# Use environment variables for S3 credentials and model path
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
S3_KEY = os.environ.get("S3_MODEL_KEY")
MODEL_PATH = "/tmp/Pretrained_model.h5"

# Fallback for local development
if not S3_BUCKET or not S3_KEY:
    print("S3 environment variables not set. Using local model path for development.")
    MODEL_PATH = "ml_model/models/Pretrained_model.h5"
    
# The rest of your code remains largely the same, but structured to run on a production server.
CLASSES = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy","Grape___Black_rot",
    "Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight",
    "Potato___Late_blight","Potato___healthy","Raspberry___healthy","Soybean___healthy",
    "Squash___Powdery_mildew","Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

PLANT_PREFIX = {
    "tomato": "Tomato___",
    "potato": "Potato___",
    "maize": "Corn_(maize)___",
    "corn": "Corn_(maize)___",
    "grape": "Grape___",
    "apple": "Apple___",
    "pepper": "Pepper,_bell___",
    "cherry": "Cherry_(including_sour)___",
    "blueberry": "Blueberry___",
    "peach": "Peach___",
    "raspberry": "Raspberry___",
    "soybean": "Soybean___",
    "squash": "Squash___",
    "strawberry": "Strawberry___",
    "orange": "Orange___",
}

INPUT_SIZE = 224

def build_model(weights_path=None, num_classes=38):
    base_model = InceptionResNetV2(include_top=False, weights="imagenet",
                                   input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_average_pooling2d_1")(x)
    x = Dropout(0.5, name="dropout_3")(x)
    x = Dense(1024, activation="relu", name="dense_4")(x)
    x = Dropout(0.5, name="dropout_4")(x)
    x = Dense(512, activation="relu", name="dense_5")(x)
    x = Dropout(0.5, name="dropout_5")(x)
    x = Dense(256, activation="relu", name="dense_6")(x)
    outputs = Dense(num_classes, activation="softmax", name="dense_7")(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    if weights_path:
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    out_dim = model.output_shape[-1]
    if out_dim != len(CLASSES):
        raise RuntimeError(f"Model head has {out_dim} outputs, but {len(CLASSES)} classes are defined. "
                           f"Fix weights or class list before serving.")
    return model

def preprocess_image(file, target_size=(INPUT_SIZE, INPUT_SIZE)):
    img = Image.open(file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)

def _softmax_safe(vec):
    vec = np.asarray(vec, dtype="float64")
    vec = vec - np.max(vec)
    e = np.exp(vec)
    s = e.sum()
    return e / s if s > 0 else np.full_like(vec, 1.0 / len(vec))

def predict_image(model, image_array, plant_name=None):
    raw = model.predict(image_array)
    probs = np.squeeze(raw)
    if probs.ndim != 1 or probs.size != len(CLASSES):
        return {
            "error": f"Unexpected prediction shape {probs.shape}; expected ({len(CLASSES)},).",
            "predicted_class": None,
            "confidence": None
        }

    if plant_name:
        prefix = PLANT_PREFIX.get(plant_name.lower())
        idxs = [i for i, c in enumerate(CLASSES) if prefix and c.startswith(prefix)]
        if idxs:
            subset = np.clip(probs[idxs], 1e-12, 1.0)
            subset = _softmax_safe(np.log(subset))
            k = int(np.argmax(subset))
            top_idx = idxs[k]
            return {
                "predicted_class": CLASSES[top_idx],
                "confidence": float(subset[k]),
                "filtered_predictions": {CLASSES[i]: float(p) for i, p in zip(idxs, subset)}
            }

    top_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASSES[top_idx],
        "confidence": float(probs[top_idx])
    }

application = Flask(__name__)
CORS(application)

# Download model from S3 if S3 environment variables are set
if S3_BUCKET and S3_KEY:
    try:
        s3 = boto3.client('s3')
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        s3.download_file(S3_BUCKET, S3_KEY, MODEL_PATH)
        print(f"Model downloaded successfully from S3://{S3_BUCKET}/{S3_KEY}")
    except Exception as e:
        print(f"Error downloading model from S3: {e}")
        sys.exit(1)

model = build_model(weights_path=MODEL_PATH, num_classes=len(CLASSES))

@application.route("/predict", methods=["POST"])
def predict_api():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    plant = request.form.get("plant", type=str)
    img_arr = preprocess_image(file)
    res = predict_image(model, img_arr, plant)
    if res.get("error"):
        return jsonify(res), 500
    return jsonify(res)

def topk_from_probs(probs, k=5):
    idxs = np.argsort(probs)[-k:][::-1]
    return [{"class": CLASSES[i], "confidence": float(probs[i])} for i in idxs]

def topk_from_probs_subset(probs, idxs, k=5):
    subset = probs[idxs]
    subset = _softmax_safe(subset)
    topk_idx = np.argsort(subset)[-min(5, len(subset)):][::-1]
    return [
        {"class": CLASSES[idxs[i]], "confidence": float(subset[i])}
        for i in topk_idx
    ]

@application.route("/topk", methods=["POST"])
def topk_api():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    plant = request.form.get("plant", type=str)

    img_arr = preprocess_image(file)
    raw = model.predict(img_arr)
    probs = np.squeeze(raw)

    if probs.ndim != 1 or probs.size != len(CLASSES):
        return jsonify({"error": f"Unexpected prediction shape {probs.shape}; expected ({len(CLASSES)},)."}), 500

    if plant:
        prefix = PLANT_PREFIX.get(plant.lower())
        if not prefix:
            return jsonify({"error": f"Unknown plant '{plant}'"}), 400

        idxs = [i for i, c in enumerate(CLASSES) if c.startswith(prefix)]
        if not idxs:
            return jsonify({"error": f"No classes found for plant '{plant}'"}), 400

        subset_probs = probs[idxs]
        subset_probs = _softmax_safe(subset_probs)
        topk_idx = np.argsort(subset_probs)[-min(5, len(subset_probs)):][::-1]

        top5 = [
            {"class": CLASSES[idxs[i]], "confidence": float(subset_probs[i])}
            for i in topk_idx
        ]
        return jsonify({"topk": top5})
    top5 = topk_from_probs(probs)
    return jsonify({"topk": top5})

# Elastic Beanstalk automatically runs gunicorn, so no need for `if __name__ == "__main__"`

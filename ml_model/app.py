# Standard library imports
import os
import io
import sys
from collections import OrderedDict

# Flask and web-related imports
from flask import Flask, request, jsonify
from flask_cors import CORS

# Deep learning framework imports
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm

# AWS SDK
import boto3

# --- Environment variables ---
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
S3_KEY = os.environ.get("S3_MODEL_KEY")
MODEL_PATH = "/tmp/pretrained_model.pth"

# Model configuration
INPUT_SIZE = 224
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

# Pre-processing transformations
val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

model = None
device = torch.device("cpu")

# --- Helper functions ---
def download_model_from_s3(bucket, key, path):
    """Download the model from S3 to local path."""
    try:
        s3 = boto3.client("s3")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        s3.download_file(bucket, key, path)
        print(f"Model downloaded successfully from S3://{bucket}/{key}")
    except Exception as e:
        print(f"Error downloading model from S3: {e}")
        sys.exit(1)

def load_model_weights(model, weights_path):
    """Load weights, handling DataParallel 'module.' prefix if necessary."""
    state_dict = torch.load(weights_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove 'module.' if exists
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def build_model(weights_path=None, num_classes=38):
    global model
    if model is None:
        print("Loading model...")
        model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
        if weights_path and os.path.exists(weights_path):
            model = load_model_weights(model, weights_path)
            print(f"Model weights loaded from {weights_path}")
        else:
            print("Warning: Model weights not found, using untrained model")
        model.to(device)
        model.eval()
    return model

def preprocess_image(file):
    img = Image.open(file).convert('RGB')
    return val_transform(img).unsqueeze(0)

def topk_from_probs(probs, k=5):
    top_k_values, top_k_indices = torch.topk(probs, k)
    return [{"class": CLASSES[i], "confidence": float(v)} for v, i in zip(top_k_values, top_k_indices)]

# --- Flask app ---
application = Flask(__name__)
CORS(application)

# Ensure model is present
if not os.path.exists(MODEL_PATH):
    if S3_BUCKET and S3_KEY:
        download_model_from_s3(S3_BUCKET, S3_KEY, MODEL_PATH)
    else:
        print("S3 environment variables not set. Cannot download model.")
        sys.exit(1)
else:
    print(f"Using existing model at {MODEL_PATH}")

# --- API routes ---
@application.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running"})

@application.route("/predict", methods=["POST"])
def predict_api():
    global model
    if model is None:
        model = build_model(weights_path=MODEL_PATH, num_classes=len(CLASSES))

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    plant = request.form.get("plant", type=str)

    inputs = preprocess_image(file).to(device)
    outputs = model(inputs)
    probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu()

    if plant:
        prefix = PLANT_PREFIX.get(plant.lower())
        if prefix:
            idxs = [i for i, c in enumerate(CLASSES) if c.startswith(prefix)]
            if idxs:
                subset_probs = probs[idxs]
                pred_idx = idxs[torch.argmax(subset_probs).item()]
                confidence = float(subset_probs.max().item())
                return jsonify({"prediction": CLASSES[pred_idx], "confidence": confidence})

    pred_idx = torch.argmax(probs).item()
    confidence = float(probs.max().item())
    return jsonify({"prediction": CLASSES[pred_idx], "confidence": confidence})

@application.route("/topk", methods=["POST"])
def topk_api():
    global model
    if model is None:
        model = build_model(weights_path=MODEL_PATH, num_classes=len(CLASSES))

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    plant = request.form.get("plant", type=str)

    inputs = preprocess_image(file).to(device)
    outputs = model(inputs)
    probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu()

    topk = []
    if plant:
        prefix = PLANT_PREFIX.get(plant.lower())
        if prefix:
            idxs = [i for i, c in enumerate(CLASSES) if c.startswith(prefix)]
            subset_probs = probs[idxs]
            top_k_values, top_k_indices = torch.topk(subset_probs, k=min(5, len(subset_probs)))
            topk = [{"class": CLASSES[idxs[i]], "confidence": float(v)} for v, i in zip(top_k_values, top_k_indices)]
        else:
            topk = topk_from_probs(probs)
    else:
        topk = topk_from_probs(probs)

    return jsonify({"topk": topk})

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    print("Starting Flask API on port 5001...")
    model = build_model(weights_path=MODEL_PATH, num_classes=len(CLASSES))
    application.run(host="0.0.0.0", port=5001, debug=True)

import os
import io
import json
import base64
import requests
import numpy as np
import onnxruntime as ort
from PIL import Image

# --- Classes ---
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

# --- Model ---
INPUT_SIZE = 224
ONNX_PATH = os.environ.get("ONNX_MODEL_PATH", "/opt/models/pretrained_model.onnx")
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# --- Helpers ---
def preprocess(img_pil):
    img = img_pil.resize((INPUT_SIZE, INPUT_SIZE)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr.astype(np.float32)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# --- Lambda Handler ---
def lambda_handler(event, context):
    try:
        body = event.get("body")
        if not body:
            return {"statusCode": 400, "body": json.dumps({"error": "No body provided"})}

        if event.get("isBase64Encoded", False):
            body = base64.b64decode(body).decode("utf-8")

        data = json.loads(body)

        # --- Load Image ---
        if "image" in data:
            img_bytes = base64.b64decode(data["image"])
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif "image_url" in data:
            resp = requests.get(data["image_url"])
            pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            return {"statusCode": 400, "body": json.dumps({"error": "No image provided"})}

        plant = data.get("plant", "")
        arr = preprocess(pil)
        logits = sess.run(None, {"input": arr})[0][0]
        probs = softmax(logits)

        # --- Prediction ---
        mode = data.get("mode", "predict")
        if plant:
            prefix = PLANT_PREFIX.get(plant.lower())
            if prefix:
                idxs = [i for i, c in enumerate(CLASSES) if c.startswith(prefix)]
                sub = probs[idxs]
                if mode == "topk":
                    order = np.argsort(-sub)[:5]
                    topk = [{"class": CLASSES[idxs[k]], "confidence": float(sub[k])} for k in order]
                    return {"statusCode": 200, "body": json.dumps({"topk": topk})}
                else:
                    j = int(np.argmax(sub))
                    return {"statusCode": 200, "body": json.dumps({"prediction": CLASSES[idxs[j]], "confidence": float(sub[j])})}

        if mode == "topk":
            order = np.argsort(-probs)[:5]
            topk = [{"class": CLASSES[k], "confidence": float(probs[k])} for k in order]
            return {"statusCode": 200, "body": json.dumps({"topk": topk})}

        i = int(np.argmax(probs))
        return {"statusCode": 200, "body": json.dumps({"prediction": CLASSES[i], "confidence": float(probs[i])})}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

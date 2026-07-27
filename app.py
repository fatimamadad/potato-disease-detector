from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import base64
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIGURATION
# ============================================
IMAGE_SIZE = 224
UPLOAD_FOLDER = "/tmp"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Groq API configuration from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = os.environ.get("GROQ_API_URL", "")

class_names = ["Early_blight", "healthy", "late_blight"]

# ============================================
# LOAD TFLITE MODEL FROM HUGGING FACE
# ============================================
interpreter = None
input_details = None
output_details = None

REPO_ID = "nainhaider/Potato_Disease_Dection_model"
MODEL_FILENAME = "potato_disease_detection.tflite"

try:
    print("=" * 60)
    print("Downloading TFLite model from Hugging Face...")
    print("=" * 60)

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME
    )
    print(f"✓ Model cached at: {model_path}")

    print("Loading TFLite interpreter...")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✓ Model loaded successfully!")

except Exception as e:
    print("❌ Error loading model from Hugging Face:")
    print(e)

# ============================================
# HELPER FUNCTIONS
# ============================================
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32)

    # ResNet50-style preprocessing (BGR + mean subtraction), matching
    # tensorflow.keras.applications.resnet50.preprocess_input
    img_array = img_array[..., ::-1]  # RGB -> BGR
    mean = [103.939, 116.779, 123.68]
    img_array[..., 0] -= mean[0]
    img_array[..., 1] -= mean[1]
    img_array[..., 2] -= mean[2]

    return np.expand_dims(img_array, axis=0)

def predict(img_path):
    try:
        img_array = preprocess_image(img_path)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)
        return predicted_class, round(confidence, 2)
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0

# ============================================
# ROUTES
# ============================================

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

@app.route("/", methods=["GET", "POST"])
def home():
    # Handle GET request - render the page with API keys
    if request.method == "GET":
        return render_template(
            "index.html",
            GROQ_API_KEY=GROQ_API_KEY,
            GROQ_API_URL=GROQ_API_URL
        )

    # Handle POST request - file upload for disease detection
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    if interpreter is None:
        return jsonify({"error": "Model not loaded"}), 500

    predicted_class, confidence = predict(filepath)
    if predicted_class is None:
        return jsonify({"error": "Prediction failed"}), 500

    # Encode image as base64 so the frontend can display it
    # without relying on a persistent file path (Vercel's /tmp is ephemeral)
    with open(filepath, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

    ext = filename.rsplit(".", 1)[1].lower()
    mime_type = "image/png" if ext == "png" else "image/jpeg"

    return jsonify({
        "success": True,
        "predicted_label": predicted_class,
        "confidence": confidence,
        "image_data": f"data:{mime_type};base64,{encoded_image}"
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model_loaded": interpreter is not None,
        "classes": class_names,
        "groq_configured": bool(GROQ_API_KEY and GROQ_API_URL)
    })

# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("Potato Disease Detection Server")
    print("=" * 60)
    print(f"Model Loaded: {interpreter is not None}")
    print(f"Upload Folder: {UPLOAD_FOLDER}")
    print(f"Groq API Configured: {bool(GROQ_API_KEY and GROQ_API_URL)}")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIGURATION
# ============================================
IMAGE_SIZE = 224
UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Groq API configuration from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = os.environ.get("GROQ_API_URL", "")

class_names = ["Early_blight", "healthy", "late_blight"]

# ============================================
# LOAD MODEL FROM HUGGING FACE
# ============================================
model = None
REPO_ID = "nainhaider/Potato_Disease_Dector"
MODEL_FILENAME = "potato_resnet_model.keras"

try:
    print("=" * 60)
    print("Downloading model from Hugging Face...")
    print("=" * 60)
    
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME
    )
    print(f"✓ Model cached at: {model_path}")
    
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
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

def predict(img_path):
    try:
        img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
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

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    predicted_class, confidence = predict(filepath)
    if predicted_class is None:
        return jsonify({"error": "Prediction failed"}), 500

    return jsonify({
        "success": True,
        "predicted_label": predicted_class,
        "confidence": confidence,
        "image_path": filepath
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
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
    print(f"Model Loaded: {model is not None}")
    print(f"Upload Folder: {UPLOAD_FOLDER}")
    print(f"Groq API Configured: {bool(GROQ_API_KEY and GROQ_API_URL)}")
    print("=" * 60)
    
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)

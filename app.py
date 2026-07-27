from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# New import for Hugging Face
from huggingface_hub import hf_hub_download

app = Flask(__name__)
CORS(app)

IMAGE_SIZE = 224
UPLOAD_FOLDER = "static"
# Remove local MODEL_PATH, we'll use a temporary file from HF
# MODEL_PATH = os.path.join("models", "potato_resnet_model.keras")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class_names = ["Early_blight", "healthy", "late_blight"]

model = None

# Hugging Face repo details
REPO_ID = "nainhaider/Potato_Disease_Dector"
MODEL_FILENAME = "potato_resnet_model.keras"   # <-- adjust if different

try:
    print("Downloading model from Hugging Face...")
    # This downloads the file to the local cache (~/.cache/huggingface/hub)
    # and returns the path to the cached file.
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME
    )
    print(f"Model downloaded to: {model_path}")
    
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")

except Exception as e:
    print("❌ Error loading model from Hugging Face:")
    print(e)
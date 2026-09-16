#!/usr/bin/env python3
"""
Pneumonia Chest X-Ray Classification - Gradio Web Application
Interactive Clinical AI Diagnostic Screening Assistant.
"""

import os
import sys
import socket
from pathlib import Path
import numpy as np
import cv2

# Ensure UTF-8 output on Windows consoles
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

# Optional dotenv loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr
import tensorflow as tf
from tensorflow import keras

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "best_pneumonia_model.keras")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", 7860))
SHARE = os.environ.get("SHARE", "false").lower() in ("true", "1", "yes")
IMG_SIZE = (224, 224)

# -----------------------------------------------------------------------------
# Model Loader
# -----------------------------------------------------------------------------
if not os.path.isfile(MODEL_PATH):
    print(f"[FATAL] Model file '{MODEL_PATH}' was not found.")
    print("Please make sure the trained model file is in the project directory.")
    sys.exit(1)

print(f"Loading deep learning model from '{MODEL_PATH}'...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# -----------------------------------------------------------------------------
# Helper: Find Sample Images for Examples UI
# -----------------------------------------------------------------------------
sample_examples = []
test_dir = Path("test")
if test_dir.exists():
    normal_samples = list((test_dir / "NORMAL").glob("*.jpeg")) + list((test_dir / "NORMAL").glob("*.jpg"))
    pneumonia_samples = list((test_dir / "PNEUMONIA").glob("*.jpeg")) + list((test_dir / "PNEUMONIA").glob("*.jpg"))
    if normal_samples:
        sample_examples.append(str(normal_samples[0]))
    if pneumonia_samples:
        sample_examples.append(str(pneumonia_samples[0]))


# -----------------------------------------------------------------------------
# Prediction Pipeline
# -----------------------------------------------------------------------------
def classify_radiograph(input_img):
    """
    Inference endpoint for Gradio Web UI.
    Takes uploaded image and returns:
    - Preview image
    - Diagnostic string
    - Confidence string
    - Probability dictionary
    """
    if input_img is None:
        return None, "Awaiting image upload", "N/A", {}

    # Convert to RGB numpy array if needed
    if not isinstance(input_img, np.ndarray):
        img_array = np.array(input_img.convert("RGB"))
    else:
        img_array = input_img

    # Handle grayscale or 4-channel images
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Resize to MobileNetV2 input resolution
    img_resized = cv2.resize(img_array, IMG_SIZE)
    img_batch = np.expand_dims(img_resized, axis=0).astype(np.float32)

    # Forward pass
    prob = float(model.predict(img_batch, verbose=0)[0][0])

    if prob >= 0.5:
        prediction = "Pneumonia"
        confidence = prob
    else:
        prediction = "Normal"
        confidence = 1.0 - prob

    pred_str = f"Diagnosis: {prediction.upper()}"
    conf_str = f"{confidence * 100:.2f}%"

    prob_dict = {
        "Pneumonia": round(prob, 4),
        "Normal": round(1.0 - prob, 4)
    }

    return img_array, pred_str, conf_str, prob_dict


def get_available_port(preferred_port=7860, max_attempts=50):
    """Finds an open localhost port starting from preferred_port."""
    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((HOST, port)) != 0:
                return port
    return preferred_port


# -----------------------------------------------------------------------------
# Gradio Interface Definition
# -----------------------------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(title="Pneumonia Chest X-Ray AI Classifier", theme=theme) as demo:
    gr.Markdown(
        """
        # 🩺 Pneumonia Chest X-Ray Diagnostic Assistant
        ### Deep Learning Radiograph Screening Powered by MobileNetV2 Transfer Learning
        Upload a standard anterior-posterior (AP/PA) chest radiograph to evaluate signs of **Pneumonia** versus **Normal** healthy lung parenchyma.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            xray_input = gr.Image(
                type="numpy",
                label="Upload Chest Radiograph (JPEG/PNG)",
                sources=["upload", "clipboard"]
            )
            run_btn = gr.Button("🔍 Run AI Diagnostic Screening", variant="primary", size="lg")

            if sample_examples:
                gr.Examples(
                    examples=sample_examples,
                    inputs=xray_input,
                    label="Representative Benchmark Radiographs (Click to Test)"
                )

        with gr.Column(scale=1):
            image_preview = gr.Image(label="Processed Radiograph Preview", interactive=False)
            with gr.Row():
                pred_output = gr.Textbox(label="Diagnostic Classification", placeholder="Pending analysis...", interactive=False)
                conf_output = gr.Textbox(label="Confidence Level", placeholder="Pending analysis...", interactive=False)
            prob_meter = gr.Label(num_top_classes=2, label="Class Probability Breakdown")

    run_btn.click(
        fn=classify_radiograph,
        inputs=xray_input,
        outputs=[image_preview, pred_output, conf_output, prob_meter]
    )

    gr.Markdown(
        """
        ---
        **Benchmarked Performance**:
        - **Test Sensitivity (Recall for Pneumonia)**: `95.13%` (371 / 390 cases identified)
        - **Test ROC-AUC**: `0.9247` | **Overall Accuracy**: `83.01%`

        *⚠️ **Medical & Ethical Disclaimer**: This deep learning software is provided exclusively for educational, research, and benchmarking demonstration purposes. It does not constitute medical advice, diagnosis, or clinical prescription. Always consult a qualified radiologist or physician for clinical evaluations.*
        """
    )


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    final_port = get_available_port(PORT)
    print(f"Launching Gradio app on {HOST}:{final_port} (share={SHARE})...")
    demo.launch(
        server_name=HOST,
        server_port=final_port,
        share=SHARE,
        prevent_thread_lock=False
    )

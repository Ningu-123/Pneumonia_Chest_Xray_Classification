#!/usr/bin/env python3
"""
Pneumonia Chest X-Ray Classification - CLI Prediction Utility
Run inference on single radiographs using the pre-trained Keras model.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np

# Suppress TensorFlow verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure UTF-8 output on Windows consoles
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

DEFAULT_MODEL_PATH = os.environ.get('MODEL_PATH', 'best_pneumonia_model.keras')
IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = float(os.environ.get('PREDICTION_THRESHOLD', 0.5))


def load_classifier(model_path: str):
    """Loads and returns the trained Keras model."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model weights file not found at: '{model_path}'. "
            "Please ensure 'best_pneumonia_model.keras' exists or specify --model <path>."
        )
    import tensorflow as tf
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    return model


def preprocess_image(image_path: str):
    """Loads and preprocesses an image for MobileNetV2 input."""
    import cv2
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found at: '{image_path}'")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read or decode image file at: '{image_path}'")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_batch = np.expand_dims(img_resized, axis=0).astype(np.float32)
    return img_batch


def run_prediction(image_path: str, model_path: str = DEFAULT_MODEL_PATH, threshold: float = DEFAULT_THRESHOLD):
    """Performs inference on a single chest X-ray image."""
    start_time = time.time()
    
    model = load_classifier(model_path)
    img_batch = preprocess_image(image_path)
    
    # Forward pass
    pred_prob = float(model.predict(img_batch, verbose=0)[0][0])
    elapsed_ms = (time.time() - start_time) * 1000

    if pred_prob >= threshold:
        predicted_class = "PNEUMONIA"
        confidence = pred_prob
    else:
        predicted_class = "NORMAL"
        confidence = 1.0 - pred_prob

    result = {
        "image": str(Path(image_path).resolve()),
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "probabilities": {
            "PNEUMONIA": round(pred_prob, 4),
            "NORMAL": round(1.0 - pred_prob, 4)
        },
        "threshold_used": threshold,
        "inference_latency_ms": round(elapsed_ms, 2),
        "status": "SUCCESS"
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Classify Chest Radiographs for Pneumonia (Clinical AI Screening Assistant)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to input chest X-ray image (JPEG / PNG)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained Keras model file (.keras)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Decision threshold probability for Pneumonia classification"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format for programmatic parsing"
    )

    args = parser.parse_args()

    try:
        res = run_prediction(args.image, args.model, args.threshold)
        
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("\n" + "=" * 60)
            print("     🩺 CHEST X-RAY CLASSIFICATION RESULT")
            print("=" * 60)
            print(f"  Input File         : {res['image']}")
            print(f"  Diagnostic Result  : {res['predicted_class']}")
            print(f"  Confidence         : {res['confidence_percentage']}")
            print(f"  P(PNEUMONIA)       : {res['probabilities']['PNEUMONIA'] * 100:.2f}%")
            print(f"  P(NORMAL)          : {res['probabilities']['NORMAL'] * 100:.2f}%")
            print(f"  Latency            : {res['inference_latency_ms']:.1f} ms")
            print("=" * 60)
            if res['predicted_class'] == "PNEUMONIA":
                print("  ⚠️  Positive indicator for Pneumonia. Further clinical review recommended.")
            else:
                print("  ✅ Clear radiograph with no prominent consolidation detected.")
            print("=" * 60 + "\n")
            
    except Exception as exc:
        if args.json:
            print(json.dumps({"status": "ERROR", "message": str(exc)}, indent=2))
        else:
            print(f"\n[ERROR] Inference failed: {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

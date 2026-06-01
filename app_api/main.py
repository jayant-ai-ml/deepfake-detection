"""HTTP API for mobile deepfake video predictions."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

# MediaPipe imports TensorFlow when it is installed. The legacy TensorFlow 2.10
# environment needs protobuf's Python implementation to coexist with MediaPipe.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from utils.landmark_extractor import MediaPipeLandmarkExtractor

BASE_DIR = Path(__file__).resolve().parents[1]
DEPLOYMENT_MODEL_DIR = Path(
    os.environ.get("DEPLOYMENT_MODEL_DIR", BASE_DIR / "data" / "processed")
)
MODEL_PATH = DEPLOYMENT_MODEL_DIR / "enhanced_ensemble_model.pkl"
LANDMARK_SCALER_PATH = DEPLOYMENT_MODEL_DIR / "scaler.pkl"
DESKTOP_APP_PATH = BASE_DIR / "app_api" / "static" / "index.html"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MIN_VALID_FRAMES = 9
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="Deepfake Detector API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_artifact: dict | None = None
_extractor: MediaPipeLandmarkExtractor | None = None
_landmark_scaler = None


def get_artifact() -> dict:
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Trained model not found: {MODEL_PATH}")
        with MODEL_PATH.open("rb") as model_file:
            _artifact = pickle.load(model_file)
    return _artifact


def get_extractor() -> MediaPipeLandmarkExtractor:
    global _extractor
    if _extractor is None:
        _extractor = MediaPipeLandmarkExtractor()
    return _extractor


def get_landmark_scaler():
    global _landmark_scaler
    if _landmark_scaler is None:
        if not LANDMARK_SCALER_PATH.exists():
            raise RuntimeError(f"Landmark scaler not found: {LANDMARK_SCALER_PATH}")
        with LANDMARK_SCALER_PATH.open("rb") as scaler_file:
            _landmark_scaler = pickle.load(scaler_file)["scaler"]
    return _landmark_scaler


def normalize_landmarks(sequence: np.ndarray) -> np.ndarray:
    """Apply the frame-level normalization used during dataset preparation."""
    original_shape = sequence.shape
    normalized = get_landmark_scaler().transform(sequence.reshape(-1, original_shape[-1]))
    return normalized.reshape(original_shape).astype(np.float32)


def extract_enhanced_temporal_features(sequence: np.ndarray) -> np.ndarray:
    """Apply the same feature engineering used by final_training.py."""
    x = sequence[np.newaxis, ...]
    diff = np.diff(x, axis=1)
    acc = np.diff(diff, axis=1)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8

    features = [
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        np.percentile(x, 25, axis=1),
        np.percentile(x, 75, axis=1),
        np.median(x, axis=1),
        x[:, 0, :],
        x[:, -1, :],
        x.max(axis=1) - x.min(axis=1),
        diff.mean(axis=1),
        diff.std(axis=1),
        np.abs(diff).max(axis=1),
        acc.mean(axis=1),
        acc.std(axis=1),
        np.mean(((x - mean) / std) ** 3, axis=1),
        np.mean(((x - mean) / std) ** 4, axis=1) - 3,
    ]
    return np.concatenate(features, axis=1).astype(np.float32)


def predict_video(video_path: Path) -> dict:
    artifact = get_artifact()
    landmarks, valid_frames = get_extractor().extract_from_video(video_path, max_frames=30)
    if len(valid_frames) < MIN_VALID_FRAMES:
        raise ValueError(
            "Face clearly detect nahi hua. Front-facing aur well-lit video upload karein."
        )

    normalized_landmarks = normalize_landmarks(landmarks)
    features = extract_enhanced_temporal_features(normalized_landmarks)
    scaled_features = artifact["scaler"].transform(features)
    selector = artifact.get("selector")
    if selector is not None:
        scaled_features = selector.transform(scaled_features)

    fake_probability = float(artifact["ensemble"].predict_proba(scaled_features)[0, 1])
    policy = artifact.get("decision_policy", {})
    real_max_probability = float(policy.get("real_max_probability", 0.20))
    fake_min_probability = float(policy.get("fake_min_probability", 0.75))

    if fake_probability >= fake_min_probability:
        prediction = "deepfake"
        is_deepfake: bool | None = True
        confidence = fake_probability
    elif fake_probability <= real_max_probability:
        prediction = "real"
        is_deepfake = False
        confidence = 1.0 - fake_probability
    else:
        prediction = "review"
        is_deepfake = None
        confidence = max(fake_probability, 1.0 - fake_probability)

    return {
        "prediction": prediction,
        "is_deepfake": is_deepfake,
        "fake_probability": round(fake_probability, 4),
        "real_probability": round(1.0 - fake_probability, 4),
        "confidence": round(confidence, 4),
        "real_max_probability": round(real_max_probability, 4),
        "fake_min_probability": round(fake_min_probability, 4),
        "valid_frames": len(valid_frames),
        "sampled_frames": 30,
    }


@app.get("/", response_class=FileResponse)
def desktop_app() -> FileResponse:
    return FileResponse(DESKTOP_APP_PATH)


@app.get("/api")
def api_info() -> dict:
    return {"name": "Deepfake Detector API", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_ready": MODEL_PATH.exists() and LANDMARK_SCALER_PATH.exists(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Supported video file upload karein.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Video size 100 MB se kam rakhein.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        return predict_video(temp_path)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from preprocessor import FacePreprocessor

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
TOP_K = 3

_model: Any = None
_class_names: list[str] = []
_img_size = 48
_preprocessor: FacePreprocessor | None = None

@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    try:
        yield
    finally:
        global _model, _class_names, _preprocessor
        _model = None
        _class_names = []
        _preprocessor = None


app = FastAPI(
    title="Emotions ML API",
    description="API распознавания эмоций по лицу",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_artifacts() -> None:
    global _model, _class_names, _img_size, _preprocessor
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    classes = metadata.get("classes")
    _img_size = int(metadata.get("img_size", 48))
    _class_names = [str(name) for name in classes]
    _model = load_model(MODEL_PATH)
    _preprocessor = FacePreprocessor()


def predict_from_bytes(image_bytes: bytes) -> dict:
    if _model is None or _preprocessor is None:
        raise RuntimeError("Model or preprocessor is not loaded")
    input_tensor = _preprocessor.preprocess(image_bytes=image_bytes, img_size=_img_size)
    probs = _model.predict(input_tensor, verbose=0)[0]
    probs = probs.astype(float)

    if probs.shape[0] != len(_class_names):
        raise RuntimeError("Model output shape does not match metadata classes")

    top_indices = np.argsort(probs)[::-1][:TOP_K]
    top_predictions = [
        {"emotion": _class_names[idx], "probability": round(float(probs[idx]), 4)}
        for idx in top_indices
    ]
    best = top_predictions[0]
    return {
        "emotion": best["emotion"],
        "probability": best["probability"],
        "top_predictions": top_predictions,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Received empty image")

    try:
        return predict_from_bytes(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

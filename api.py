import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from inference import CaptionGenerator, load_model as build_model

# --- Configuration (override with environment variables) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/best_model.pth")
MODEL_REPO = os.environ.get("MODEL_REPO", "sarveshraam/image-captioner-model")
# Tried in order on the Model Hub: the exported model first, then the original training checkpoint
MODEL_FILES = os.environ.get("MODEL_FILE", "model.pth,best_model.pth").split(",")
QUANTIZE = os.environ.get("QUANTIZE", "0") == "1"
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
LOW_CONFIDENCE = float(os.environ.get("LOW_CONFIDENCE", "0.3"))

# Global generator instance
generator = None
load_error = None


def resolve_checkpoint():
    """Use a local checkpoint if present, otherwise fetch it from the Hugging Face Model Hub."""
    if os.path.exists(CHECKPOINT_PATH):
        return CHECKPOINT_PATH
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    for filename in MODEL_FILES:
        try:
            print(f"[INFO] Downloading {MODEL_REPO}/{filename}...", flush=True)
            return hf_hub_download(repo_id=MODEL_REPO, filename=filename)
        except EntryNotFoundError:
            continue
    raise FileNotFoundError(f"None of {MODEL_FILES} found in {MODEL_REPO}")


def load_model():
    global generator, load_error
    try:
        path = resolve_checkpoint()
        print(f"[INFO] Loading checkpoint from {path} on {DEVICE} (quantize={QUANTIZE})...", flush=True)
        model, vocab = build_model(path, DEVICE, quantize=QUANTIZE)
        generator = CaptionGenerator(model, vocab, DEVICE)
        load_error = None
        print(f"[INFO] Model ready ({model.encoder.encoder_type}, vocab size {len(vocab)}).", flush=True)
    except Exception as e:
        load_error = str(e)
        print(f"[ERROR] Failed to load model: {e}", flush=True)


@asynccontextmanager
async def lifespan(app):
    load_model()
    yield


app = FastAPI(title="Image Captioning API", lifespan=lifespan)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def low_detail(image):
    """Nearly uniform images (blank, solid colour, heavily over/under-exposed) give hallucinated captions."""
    gray = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32)
    return float(gray.std()) < 10.0


@app.get("/")
def root():
    return {"service": "image-captioning", "endpoints": ["/health", "/predict"]}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": generator is not None,
        "encoder": generator.model.encoder.encoder_type if generator else None,
        "error": load_error,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    beam_size: int = Query(3, ge=1, le=10),
    num_captions: int = Query(3, ge=1, le=5),
):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Please retry shortly.")

    contents = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image is larger than {MAX_UPLOAD_MB} MB.")

    try:
        # exif_transpose: phone photos are often stored sideways with an orientation tag
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    # Beam search is CPU-bound, so run it off the event loop to keep /health responsive
    start = time.perf_counter()
    result = await run_in_threadpool(generator.describe, image, beam_size, num_captions)
    elapsed_ms = round((time.perf_counter() - start) * 1000)

    warnings = []
    if low_detail(image):
        warnings.append("This image has very little visual detail, so the caption is probably unreliable.")
    elif result["confidence"] < LOW_CONFIDENCE:
        warnings.append("The model is unsure about this image; the caption may be inaccurate.")

    return {
        "caption": result["captions"][0]["text"] or "No caption generated.",
        "captions": result["captions"],
        "confidence": result["confidence"],
        "warnings": warnings,
        "attention": result["attention"],
        "inference_ms": elapsed_ms,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8001")))

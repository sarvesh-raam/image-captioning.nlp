# image-captioning-nlp

[![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?logo=vercel)](https://image-captioning-nlp.vercel.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Backend-FFD21E?logo=huggingface&logoColor=000)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)]()

A full-stack image captioning service. A frozen vision backbone (CLIP ViT-B/16 or ResNet50) feeds a
Transformer encoder head and a 6-layer Transformer decoder, trained on MS COCO with cross-entropy and then
self-critical sequence training (SCST). The FastAPI backend runs in Docker on Hugging Face Spaces; the
React/Vite frontend is hosted on Vercel and shows per-word attention heatmaps, alternative captions and
model confidence.

## Quickstart

### Frontend
```bash
cd webapp
npm install
npm run dev
```

### Backend
```bash
pip install -r requirements.txt
python api.py
```
Uses `checkpoints/best_model.pth` if present (override with `CHECKPOINT_PATH`), otherwise downloads
`model.pth` / `best_model.pth` from the
[`sarveshraam/image-captioner-model`](https://huggingface.co/sarveshraam/image-captioner-model) Model Hub repo.
Set `QUANTIZE=1` for int8 inference on CPU.

## Training pipeline

```bash
pip install -r requirements-train.txt

# 1. COCO Karpathy split (113k train / 5k val / 5k test images, 5 captions each) + vocabulary
python prepare_data.py                      # full dataset, ~19 GB of images
python prepare_data.py --max-train 20000    # or a smaller subset

# 2. Run the frozen backbone once and cache its features (~37 GB for the full set with CLIP B/16)
python extract_features.py --encoder clip-vit-b16

# 3. Cross-entropy training, then SCST (optimises CIDEr directly). Both resume automatically and
#    select the best epoch by validation CIDEr.
python train.py --encoder clip-vit-b16
python train.py --encoder clip-vit-b16 --scst

# 4. COCO metrics on the Karpathy test split (+ beam size / repetition penalty ablations)
python evaluate.py --checkpoint checkpoints/clip-vit-b16_scst_best.pth --beam-sizes 1 3 5 --ablate-penalty

# 5. Shrink for deployment (drops optimizer state, float16) and deploy
python export_model.py checkpoints/clip-vit-b16_scst_best.pth checkpoints/model.pth
python deploy_to_hf.py
```

Training on cached features only runs the encoder head and decoder, so it is fast on a single GPU
(a free Kaggle/Colab T4 is enough). METEOR and SPICE are reported only when Java is installed.
Per-epoch metrics are written to `checkpoints/<encoder>_history.json`, evaluation results to `results/`.

### Tests
```bash
python -m pytest -q tests
```
CI (`.github/workflows/ci.yml`) runs the tests plus the frontend lint and build on every push.

## Deployment

**Frontend (Vercel)**
Pushes to the `main` branch are automatically deployed by Vercel. Ensure the `VITE_API_URL` environment variable is set to the Hugging Face Space URL.

**Backend (Hugging Face / Docker)**
The API is containerized using the provided `Dockerfile`; the model checkpoint is downloaded from the
Model Hub at build time so cold starts stay fast. Deploy updates via the CLI:
```bash
python deploy_to_hf.py
```

## API Reference

**`POST /predict`**

Accepts an image (max 10 MB) and optional `beam_size` (1-10, default 3) and `num_captions` (1-5, default 3)
query parameters.
Errors are returned with proper status codes and a `detail` message
(`400` invalid image, `413` too large, `503` model still loading).

```bash
curl -X POST -F "file=@image.jpg" https://your-space-url.hf.space/predict
```
```json
{
  "caption": "A person riding a bicycle on a city street.",
  "captions": [{"text": "A person riding a bicycle on a city street.", "score": -0.41}, "..."],
  "confidence": 0.47,
  "warnings": [],
  "attention": {"grid": 14, "crop": "center", "words": ["a", "person", "..."], "word_probs": [0.8, "..."],
                "maps": [["196 weights per word"]]},
  "inference_ms": 850
}
```
`attention.maps[i]` is the decoder's cross-attention over the `grid x grid` image patches for word `i`.
`warnings` flags images with almost no visual detail (where any caption is a guess) or low model confidence.

**`GET /health`** returns `{"status": "ok", "model_loaded": true, "encoder": "resnet50"}`.

> The backend runs on a free Hugging Face Space, which sleeps after inactivity. The first request after
> a sleep can take a minute or two; the frontend pings `/health` on load and retries automatically.

## Project Structure

- `webapp/` - React/Vite frontend (`AttentionOverlay.jsx` draws the heatmaps)
- `api.py` - FastAPI entrypoint
- `model.py` - model architecture (pluggable backbone, Transformer head and decoder)
- `inference.py` - checkpoint loading, beam search, attention extraction
- `prepare_data.py`, `extract_features.py` - dataset and feature cache
- `train.py` - cross-entropy and SCST training on cached features
- `evaluate.py`, `metrics.py` - COCO metrics and CIDEr-D
- `export_model.py` - deployment checkpoint export
- `app.py` - optional Gradio demo
- `Dockerfile`, `deploy_to_hf.py` - backend deployment

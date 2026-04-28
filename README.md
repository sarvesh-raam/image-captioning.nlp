# image-captioning-nlp

[![Vercel](https://img.shields.io/badge/Vercel-Deployed-black?logo=vercel)](https://image-captioning-nlp.vercel.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Backend-FFD21E?logo=huggingface&logoColor=000)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)]()

A full-stack image captioning service. The backend is a FastAPI application serving a PyTorch ResNet50 + Transformer model, deployed via Docker on Hugging Face Spaces. The frontend is a React/Vite application hosted on Vercel.

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
*Requires `checkpoints/best_model.pth` and `vocabulary.pkl` for inference.*

## Deployment

**Frontend (Vercel)**
Pushes to the `main` branch are automatically deployed by Vercel. Ensure the `VITE_API_URL` environment variable is set to the Hugging Face Space URL.

**Backend (Hugging Face / Docker)**
The API is containerized using the provided `Dockerfile`. Deploy updates via the CLI:
```bash
python deploy_to_hf.py
```

## API Reference

**`POST /predict`**

Accepts an image and returns a generated caption.

```bash
curl -X POST -F "file=@image.jpg" https://your-space-url.hf.space/predict
```
```json
{
  "caption": "a person riding a bicycle on a city street."
}
```

## Project Structure

- `/webapp` - React/Vite frontend source
- `model.py` - PyTorch model architecture
- `api.py` - FastAPI entrypoint
- `train.py` - Model training script (MS COCO)
- `Dockerfile` - Backend container config

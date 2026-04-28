# Image Captioning NLP

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

A scalable image captioning service combining a ResNet50 vision encoder with a multi-head attention transformer decoder. Trained on the MS COCO 2014 dataset.

The project is split into a PyTorch/FastAPI backend for model inference and a React/Vite frontend for the user interface.

## System Architecture

* **Vision Backbone:** ResNet50 (pre-trained, customized projection layers)
* **Decoder:** 6-layer Transformer Decoder with 8 attention heads (512 embedding dim)
* **Inference Strategy:** Beam Search (k=3) with length normalization
* **API:** FastAPI serving REST endpoints, containerized via Docker
* **Client:** React 18 + Vite with glassmorphism UI tokens

## Setup Instructions

### 1. Backend (Inference API)

Requirements: Python 3.10+ and a CUDA-compatible GPU (optional but recommended for training).

```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the development server
python api.py
```
The API will be available at `http://localhost:8001`.

### 2. Frontend (React Client)

Requirements: Node.js 18+.

```bash
cd webapp
npm install

# Start the frontend development server
npm run dev
```

## Production Deployment

### Backend Server (Docker / Hugging Face Spaces)

The backend is configured for containerized deployment. A `deploy_to_hf.py` script is provided for automated syncing to Hugging Face Spaces.

```bash
# Automate deployment to Hugging Face Spaces
python deploy_to_hf.py
```
*Note: Ensure your `best_model.pth` is placed in the `checkpoints/` directory before deploying.*

### Frontend Application (Vercel)

The React client can be deployed to Vercel or Netlify. You must provide the production API URL during the build step.

Environment variables required:
`VITE_API_URL` - The HTTPS endpoint of your deployed FastAPI server.

## API Reference

### `POST /predict`
Generates a caption for an uploaded image.

**Request:**
* `Content-Type: multipart/form-data`
* `file`: (Required) The image binary.

**Response:**
```json
{
  "caption": "a person riding a bicycle on a city street."
}
```

## Repository Structure

```text
.
├── api.py                  # FastAPI server configuration
├── app.py                  # Fallback Gradio interface
├── data_loader.py          # COCO dataset parsing and transforms
├── deploy_to_hf.py         # Automated deployment script
├── inference.py            # Beam search and generation logic
├── model.py                # PyTorch architecture (ResNet + Transformer)
├── train.py                # Training loop and checkpointing
├── Dockerfile              # Production container definition
├── requirements.txt        # Python dependencies
└── webapp/                 # React frontend directory
```

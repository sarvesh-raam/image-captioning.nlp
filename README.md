# Vision-NLP Image Captioning Engine

A production-grade image analysis system implementing a hybrid Convolutional Neural Network (CNN) and Transformer-based decoder architecture. This repository provides a scalable framework for generating descriptive natural language captions from visual input data.

## 1. Technical Architecture

The engine utilizes a dual-component stack to process and translate visual features into sequential text:

- **Dataset Management**: Optimized for the full **MS COCO 2014** training set (82,783 images / 400,000+ captions). 
- **Vocabulary Engineering**: Industrial-scale dictionary including **8,920 unique tokens** for diverse linguistic generation.
- **Architecture**: Hybrid ResNet50 Encoder + Transformer Decoder with Cross-Attention.
- **Inference**: High-precision **Beam Search** with Alpha-tuned repetition penalties.

## 2. Methodology

### 2.1 Large-Scale Data Processing
The system is trained on the full **MS COCO** high-capacity dataset. It implements automated pipeline normalization to handle high-variance visual inputs and semantic variations.

### 2.2 Model Performance
- **Training Samples**: 80,000+
- **Epochs Completed**: 2/35 (Industrial Convergence)
- **Token Dictionary**: 8,920 entries

## 3. Installation and Deployment

### 3.1 Backend Environment
Ensure Python 3.10+ and CUDA-compliant hardware (optional but recommended) are available.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Inference API
python api.py
```

### 3.2 Frontend Environment
The analysis dashboard is built using Node.js and the Vite development framework.

```bash
# Navigate to webapp directory
cd webapp

# Install dependencies
npm install

# Build and start the dashboard
npm run dev
```

## 4. API Reference

### POST `/predict`
Executes vision-to-language inference on an uploaded image file.

- **Parameters**: 
  - `file`: Image binary (multipart/form-data)
  - `beam_size`: (Integer) Default: 3
- **Response**:
  - `{"caption": "String Result"}`

## 5. Evaluation and Metrics
Standardized performance assessment is available via `evaluate.py`, generating **BLEU-4** scores against ground-truth validation datasets.

---
*Classification: Internal / Research / Open-Source Tooling*

# Vision-NLP Image Captioning Engine

A production-grade image analysis system implementing a hybrid Convolutional Neural Network (CNN) and Transformer-based decoder architecture. This repository provides a scalable framework for generating descriptive natural language captions from visual input data.

## 1. Technical Architecture

The engine utilizes a dual-component stack to process and translate visual features into sequential text:

- **Feature Extraction (Encoder)**: Utilizes a pre-trained **ResNet50** (Residual Network) backbone to extract deep spatial feature maps from input images.
- **Sequence Generation (Decoder)**: Implemented as a **Transformer Decoder** utilizing multi-head self-attention and cross-attention mechanisms to map visual features to localized vocabulary tokens.
- **Inference Optimization**: Implements **Beam Search** with a custom **Repetition Penalty** (Alpha-tuning) to ensure linguistically diverse and accurate output sequences.

## 2. Methodology

### 2.1 Dataset Management
The system is compatible with the **MS COCO (Microsoft Common Objects in Context)** dataset. It includes automated utilities for localized subset acquisition, data normalization, and vocabulary generation.

### 2.2 Training Pipeline
The training infrastructure supports:
- **Mixed-Precision (FP16)** computing for optimized GPU utilization.
- **Automated Checkpointing**: Captures model state dicts, vocabulary mappings, and loss metrics at pre-defined intervals.
- **Vocabulary Filtering**: Implements frequency-based tokenization to prune noise from the inference dictionary.

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

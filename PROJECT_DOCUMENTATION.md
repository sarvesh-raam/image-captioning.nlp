# Vision-NLP Image Captioning System | Industrial Scale Build

This repository contains a production-grade image captioning pipeline trained on the full **MS COCO (80,000 images)** dataset. The system utilizes a state-of-the-art hybrid CNN-Transformer architecture to achieve high-fidelity visual-to-text semantic translation.

## 🏗️ Model Architecture Details
- **Encoder**: ResNet50 Residual Network (Feature Map Extraction)
- **Decoder**: Transformer Decoder (Self-Attention & Cross-Attention)
- **Vocabulary**: 8,920 unique tokens (Industrial Diversity)
- **Data Scale**: 80,000 Training Samples / 35 Epoch Schedule

## 📊 Training Specifications (V4 - High Capacity)
This version of the model represents a significant leap in descriptive power, moving beyond laboratory subsets to a full-scale knowledge base. It is capable of identifying complex interactions between objects, environments, and spatial relationships.

- **Dataset**: MS COCO 2014 (Full Train Set)
- **Batch Size**: 16
- **Optimization**: Adam (1e-4) with Beam Search Alpha-tuning.

## 🚀 Deployment Instructions
1. **Inference API**: `python api.py` processes image blobs on port 8001.
2. **Analysis Dashboard**: Run `npm run dev` in the `webapp` directory to access the executive analysis suite.

---
*Status: High-Capacity Training Active | Phase 4 Deployment*

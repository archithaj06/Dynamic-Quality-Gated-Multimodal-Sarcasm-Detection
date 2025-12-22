# DQG-MSD: Dynamic-Quality-Gated Multimodal Sarcasm Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)

Sarcasm is inherently multimodal—the sentiment of the text often contradicts the visual expression or the acoustic tone. This project implements a **Late-Fusion Quality-Gating** mechanism to resolve these conflicts.

## 🧠 Key Innovations
- **Dynamic Gating:** Instead of simple concatenation, we learn a quality logit $q$ for each modality.
- **Backbone Integration:** Uses BERT (Text), ResNet50 (Video), and Wav2Vec2 (Audio).
- **Entropy Regularization:** Penalizes "lazy" gating, forcing the model to identify the most reliable modality for each sample.

## 🛠️ Project Structure
- `models/encoders.py`: Feature extraction backbones.
- `models/fusion.py`: Gating logic and feature fusion.
- `train.py`: Training loop and model demonstration.

## 🚀 Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run demo: `python train.py`
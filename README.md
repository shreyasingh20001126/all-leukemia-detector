---
title: ALL Leukemia Detector
emoji: 🔬
colorFrom: red
colorTo: purple
sdk: docker
app_file: app.py
app_port: 8501
pinned: false
---

# Acute Lymphoblastic Leukemia Detection

Weighted Ensemble CNN using 6 pretrained models achieving 97.54% accuracy.

Published: IEEE DASA 2024 - Shreya Singh et al.

## Model Weights

The trained model weights (6 CNN architectures, ~1GB total) are not stored in this repository due to GitHub's file size limits. They are automatically downloaded at Docker build time from the Hugging Face model repo:

**[shreyasingh2000/all-leukemia-weights](https://huggingface.co/shreyasingh2000/all-leukemia-weights)**

See the `Dockerfile` for the exact download step.

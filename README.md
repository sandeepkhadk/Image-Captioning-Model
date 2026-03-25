# 🖼️ Image Captioning Model

A deep learning pipeline that automatically generates natural language captions for images. It combines a **CNN-based feature extractor (Xception)** with an **LSTM-based sequence generator** trained on the Flickr8k dataset.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [License](#license)

---

## Overview

This project implements an end-to-end image captioning system using a classic **encoder-decoder** architecture:

- The **encoder** uses a pre-trained **Xception** CNN (without the top classification layer) to extract a 2048-dimensional feature vector from each image.
- The **decoder** is an **LSTM** network that takes the image features and a partial caption sequence as input, and predicts the next word — one token at a time — until an `<end>` token is generated.

The model is trained on the **Flickr8k** dataset, which contains 8,000 images each paired with five human-written captions.

---

## Architecture

```
Image ──► Xception (pretrained, no top) ──► 2048-d feature vector
                                                     │
                                                     ▼
                                              Dense (256) ──────────────────────┐
                                                                                │
Caption Sequence ──► Embedding ──► Dropout ──► LSTM (256) ──► Dropout ──────► Add ──► Dense (vocab_size) ──► Softmax ──► Next Word
```

The image feature branch and the language model branch are merged via an **addition layer**, followed by a dense softmax output over the full vocabulary.

---

## Project Structure

```
Image-Captioning-Model/
│
├── dataset/
│   ├── Flicker8k_Dataset/       # Raw images
│   └── Flicker8k_text/          # Caption annotation files
│
├── feature_extraction.py        # Xception feature extraction + caption preprocessing
├── train_model.py               # Model definition and training loop
├── test.py                      # Inference: generate captions for new images
│
├── features.p                   # Pickled image feature vectors (generated)
├── tokenizer.p                  # Pickled Keras Tokenizer (generated)
├── descriptions.txt             # Cleaned image–caption mappings (generated)
│
├── requirements.txt             # Python dependencies
└── .gitignore
```

---

## Dataset

This project uses the **Flickr8k** dataset.

1. Download the images from: [Flickr8k Dataset](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k)
2. Download the caption annotations (text files) from the same source.
3. Place them inside the `dataset/` folder:

```
dataset/
├── Flicker8k_Dataset/     ← all .jpg images go here
└── Flicker8k_text/        ← Flickr8k.token.txt, Flickr_8k.trainImages.txt, etc.
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/sandeepkhadk/Image-Captioning-Model.git
cd Image-Captioning-Model
```

**2. Create and activate a virtual environment (recommended)**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Extract image features and preprocess captions

This script loads the Xception model, extracts a feature vector for every image in the dataset, cleans the captions, and saves both to disk.

```bash
python feature_extraction.py
```

Outputs:
- `features.p` — pickled dictionary mapping image IDs → feature arrays
- `descriptions.txt` — cleaned image–caption pairs

### Step 2 — Train the model

Loads the pre-computed features, builds the encoder-decoder model, and trains it on the Flickr8k training split.

```bash
python train_model.py
```

Outputs:
- `tokenizer.p` — fitted Keras Tokenizer saved for inference
- Trained model weights (saved during/after training)

### Step 3 — Generate captions

Run inference on a new image.

```bash
python test.py
```

Edit the image path inside `test.py` to point to your target image before running.

---

## How It Works

**Feature Extraction (`feature_extraction.py`)**

- Loads `Flickr8k.token.txt` and parses it into a dictionary of `image_id → [captions]`.
- Cleans captions: lowercasing, removing punctuation, stripping short/non-alpha tokens.
- Loads the **Xception** model (pre-trained on ImageNet, top layer removed, average pooling) and extracts a 2048-d vector per image.
- Saves features as `features.p` and cleaned captions as `descriptions.txt`.

**Training (`train_model.py`)**

- Loads the training image list (`Flickr_8k.trainImages.txt`) and matches it against the feature vectors and descriptions.
- Builds and fits a **Keras Tokenizer** over all training captions; saves it to `tokenizer.p`.
- Constructs the encoder-decoder model:
  - Image input → Dense(256) → image embedding
  - Sequence input → Embedding → Dropout → LSTM(256) → Dropout → sequence embedding
  - Both embeddings are added → Dense(256, ReLU) → Dense(vocab_size, Softmax)
- Generates training sequences with `<start>` / `<end>` tokens and trains using categorical cross-entropy.

**Inference (`test.py`)**

- Loads the saved model, tokenizer, and the target image.
- Extracts image features using the same Xception extractor.
- Generates a caption word-by-word: feeds the image vector and the growing sequence into the model until `<end>` is predicted or the maximum caption length is reached.

---

## Requirements

Key dependencies (see `requirements.txt` for pinned versions):

| Package | Version |
|---|---|
| tensorflow | 2.20.0 |
| keras | 3.13.2 |
| numpy | 2.4.2 |
| pillow | 12.1.1 |
| matplotlib | 3.10.8 |
| tqdm | 4.67.3 |
| pandas | 3.0.1 |

Python 3.8+ is recommended.

---

## License

This project is open source. Feel free to use, modify, and distribute it with attribution.

---

> Built with TensorFlow/Keras · Flickr8k Dataset · Xception + LSTM architecture

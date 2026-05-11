# CIFAKE Detector — Real vs AI-Generated Image Classification

A deep learning project that classifies images as **real** or **AI-generated (fake)**, trained on the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) — 60,000 real images (CIFAR-10) and 60,000 Stable Diffusion synthetic images.

Three models are implemented and compared, with the flagship being a novel **Hybrid CNN-ViT architecture** where a CNN backbone acts as the patch embedder for a Vision Transformer — giving the transformer semantically rich tokens instead of raw pixel patches.

---

## Demo

Upload any image to the Streamlit app and get an instant REAL / FAKE verdict with confidence scores across all three models.

```bash
streamlit run app.py
```

---

## Architecture Overview

### Model 1 — CNN (ResNet-style)
A residual CNN with four stacked ResBlocks that progressively downsample from 32×32 to a 512-dim global average pooled feature vector, followed by a two-layer classification head.

```
Image [3, 32, 32]
  → Stem (Conv + BN + ReLU)
  → ResBlock  64ch  32×32
  → ResBlock 128ch  16×16  (stride=2)
  → ResBlock 256ch   8×8   (stride=2)
  → ResBlock 512ch   4×4   (stride=2)
  → AdaptiveAvgPool → Dropout → Linear(512, 128) → Linear(128, 2)
```

### Model 2 — Vision Transformer (ViT, from scratch)
A standard ViT that slices the 32×32 image into 64 non-overlapping 4×4 patches, projects them to 192-dim token embeddings, prepends a CLS token, and passes through 8 transformer encoder blocks.

```
Image [3, 32, 32]
  → PatchEmbed (4×4 patches → 64 tokens of dim 192)
  → [CLS] + positional embeddings
  → 8× TransformerBlock (MultiheadAttention + MLP)
  → LayerNorm → CLS token → Linear head
```

### Model 3 — Hybrid CNN-ViT ⭐ (flagship)
Instead of projecting raw pixel patches, a CNN backbone first extracts a spatially-aware feature map. Each cell of that feature map becomes a transformer token — carrying local texture features, translation invariance, and hierarchical representations that the transformer then reasons over globally.

```
Image [3, 32, 32]
  → CNN Tokenizer:
      Stem + ResBlock(64) + ResBlock(128, s=2) + ResBlock(256, s=2)
      → feature map [256, 8, 8]
      → 1×1 Conv projection
      → 64 tokens of dim 256
  → [CLS] + positional embeddings
  → 6× TransformerBlock
  → LayerNorm → CLS token → Linear head
```

**Why this works better than vanilla ViT on 32×32 images:**
- Raw 4×4 pixel patches carry almost no semantic signal at this resolution — CNN features do
- The CNN handles local texture artifact detection (Stable Diffusion frequency anomalies); the transformer handles global structural reasoning — each doing what it's best at
- Architecturally related to CvT, CMT, and MobileViT — active research territory

---

## Expected Performance

| Model | Test Accuracy | Macro F1 | ROC AUC |
|---|---|---|---|
| CNN (ResNet-style) | ~92–95% | ~0.92–0.95 | ~0.97–0.99 |
| ViT (from scratch) | ~87–92% | ~0.87–0.92 | ~0.94–0.97 |
| **Hybrid (CNN + ViT)** | **~94–97%** | **~0.94–0.97** | **~0.98–0.99** |

*Results at 30 epochs on a T4 GPU. Training to 60–100 epochs pushes accuracy higher.*

---

## Project Structure

```
cifake-detector/
│
├── cifake_hybrid_detector.ipynb   # Full training notebook
├── app.py                         # Streamlit web app
├── requirements.txt               # Python dependencies
│
├── CNN_best.pt                    # Saved CNN weights     ← generated after training
├── ViT_best.pt                    # Saved ViT weights     ← generated after training
├── Hybrid_best.pt                 # Saved Hybrid weights  ← generated after training
│
└── cifake/                        # Dataset (not tracked in git)
    ├── train/
    │   ├── REAL/   (30,000 images)
    │   └── FAKE/   (30,000 images)
    └── test/
        ├── REAL/   (10,000 images)
        └── FAKE/   (10,000 images)
```

---

## Quickstart

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-username/cifake-detector.git
cd cifake-detector
pip install -r requirements.txt
```

### 2. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) and place the `cifake/` folder in the project root, or use the kagglehub API:

```python
import kagglehub
path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
```

### 3. Train the models

Open `cifake_hybrid_detector.ipynb` and run all cells. The notebook is designed for **Google Colab** (T4 GPU recommended). Training all three models takes approximately:

| GPU | Time |
|---|---|
| Colab Free (T4) | ~55–68 min (30 epochs) |
| Colab Pro (A100) | ~15–20 min (30 epochs) |
| CPU only | ~5–8 hours (not recommended) |

### 4. Save your weights

After training, the notebook includes cells to save `.pt` files to **Google Drive** (recommended for Colab) or download them directly to your machine. These are loaded by the Streamlit app — you never need to retrain.

### 5. Run the web app

Place `CNN_best.pt`, `ViT_best.pt`, and `Hybrid_best.pt` in the project root, then:

```bash
streamlit run app.py
```

---

## Notebook Structure

| Section | Description |
|---|---|
| 1 | Install & import dependencies |
| 2 | Dataset loading & augmentation |
| 3 | CNN model definition |
| 4 | ViT model definition |
| 5 | **Hybrid CNN-ViT model definition** |
| 6 | Training utilities (mixed precision, cosine LR, label smoothing) |
| 7–9 | Train CNN / ViT / Hybrid |
| 10 | Training curves — all three models |
| 11 | Final evaluation (confusion matrices, ROC curves, classification reports) |
| 12 | Qualitative inspection — misclassified samples |
| 13 | **Attention map visualisation** (Hybrid only) |
| 14 | Single image inference |
| 15 | Model comparison summary & bar chart |
| 16 | **Save / restore `.pt` weight files** |

---

## Training Details

| Hyperparameter | CNN | ViT | Hybrid |
|---|---|---|---|
| Optimiser | AdamW | AdamW | AdamW |
| Learning rate | 1e-3 | 3e-4 | 3e-4 |
| Weight decay | 1e-4 | 1e-4 | 1e-4 |
| LR schedule | CosineAnnealing | CosineAnnealing | CosineAnnealing |
| Label smoothing | 0.05 | 0.05 | 0.05 |
| Gradient clipping | 1.0 | 1.0 | 1.0 |
| Mixed precision | ✅ (AMP) | ✅ (AMP) | ✅ (AMP) |
| Batch size | 128 | 128 | 128 |

**Augmentation:** random horizontal/vertical flip, rotation (±10°), colour jitter, random crop with padding.

---

## Interpretability

The Hybrid model supports **attention map visualisation** — the transformer's CLS token attention weights are extracted from the last encoder block and overlaid as a heatmap on the input image.

This reveals an interesting pattern: Stable Diffusion-generated images tend to produce **diffuse, unfocused attention** across the image, while real photographs show **structured attention** on salient objects and boundaries. This aligns with the hypothesis that SD artifacts are distributed globally rather than localised.

---

## Streamlit App

The web app (`app.py`) supports all three models and is entirely self-contained — no imports from the notebook are needed. The model architectures are redefined inline.

**Features:**
- Upload any image (JPG, PNG, WEBP)
- Switch between CNN, ViT, and Hybrid to compare predictions
- Displays verdict, confidence %, and a real/fake probability breakdown
- Graceful error messages if weight files are missing

**Deploy to Streamlit Cloud (free):**
1. Push the repo to GitHub (include `app.py`, `requirements.txt`, and the `.pt` files)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo

> **Note:** `.pt` files can be large (~50–200MB). If they exceed GitHub's 100MB file limit, use [Git LFS](https://git-lfs.github.com/) or host them on Hugging Face Hub and load them in `app.py` via `huggingface_hub.hf_hub_download`.

---

## Dataset

**CIFAKE** by Jordan J. Bird and Ahmad Lotfi.

- 60,000 **REAL** images sourced from CIFAR-10
- 60,000 **FAKE** images generated with Stable Diffusion v1.4, matching the 10 CIFAR-10 classes
- All images are 32×32 RGB
- 50,000 / 60,000 train, 10,000 / 20,000 test split

> Bird, J.J. and Lotfi, A., 2024. *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. IEEE Access.

---

## Requirements

```
torch
torchvision
streamlit
einops
Pillow
numpy
matplotlib
seaborn
scikit-learn
```

---

## License

MIT

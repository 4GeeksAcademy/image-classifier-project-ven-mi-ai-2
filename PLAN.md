# Image Classification Challenge — Implementation Plan

**Status:** Implemented  
**Intent doc:** `image_classification.md`  
**Primary deliverable:** `src/explore.ipynb`

---

## Objective

Build a dogs-vs-cats image classifier (Asirra / Microsoft Petfinder subset) using an EfficientNet-B0 ANN, optimize with Keras callbacks, evaluate on a held-out test set, and save the model.

---

## Approach (ML Engineer)

| Step | Action | Decision |
|------|--------|----------|
| 1 | Load dataset | Use existing unzipped data under `data/raw/dogs-vs-cats/.../train` (25k JPGs) |
| 2 | Visualize & prepare | Plot first 9 dogs / 9 cats; organize `train/{dog,cat}` and `test/{dog,cat}` (80/20); stream with `ImageDataGenerator` + `flow_from_directory` |
| 3 | Build ANN | EfficientNetB0 (`weights="imagenet"`, frozen) → GAP → Dense(128) → Dropout → softmax |
| 4 | Optimize | Train head, fine-tune top blocks; Checkpoint / EarlyStopping / ReduceLR; evaluate |
| 5 | Save | Write `models/efficientnet_b0_dogs_cats.keras` |

**Why progressive loading:** Challenge allows full in-memory load only if RAM > 12 GB. Generators are cleaner, match the `trdata`/`tsdata` requirement, and scale better.

**Input size:** Challenge text asks for 200×200; EfficientNet-B0 uses `(224, 224, 3)`. Training uses **224×224**.

**Improvements:** ImageNet transfer learning, augmentation, EfficientNet `preprocess_input`, Dropout, two-phase fine-tune.

**Measured result:** test accuracy **99.06%**, F1 **0.9906** (macro/weighted), test loss **0.0294** (full 5k test set).

---

## Artifacts

| Path | Role |
|------|------|
| `src/explore.ipynb` | Step-by-step notebook with markdown + final verdict |
| `src/app.py` | Scripted train/evaluate/save pipeline |
| `models/efficientnet_b0_best.keras` | Best checkpoint during fine-tune |
| `models/efficientnet_b0_dogs_cats.keras` | Final saved model |
| `data/raw/train`, `data/raw/test` | Class-folder layout for generators |
| `.venv/` | Python 3.12 env with TensorFlow (recommended on this machine) |

---

## How to run

### 1) Environment (once)

```powershell
cd c:\Work\Git\DataScince\image-classifier-project-ven-mi-ai-2
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ipykernel install --user --name=img-clf --display-name="Python (img-clf)"
```

> Use **Python 3.12** (not 3.14). System Python 3.14 had TensorFlow DLL / App Control issues on this host.

### 2) Dataset

If `data/raw/dogs-vs-cats` is missing:

1. Download: https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip  
2. Unzip into `data/raw/dogs-vs-cats/` so images are at  
   `data/raw/dogs-vs-cats/dogs-vs-cats/train/*.jpg`

Class folders are created automatically by the notebook / `app.py`.

### 3) Jupyter notebook (recommended)

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook src/explore.ipynb
```

Select kernel **Python (img-clf)** (or the `.venv` interpreter). Run all cells top → bottom.

### 4) Scripted training

```powershell
.\.venv\Scripts\Activate.ps1
python src/app.py
```

### 5) Expected outputs

- Plots of sample dogs/cats  
- Model summary + Phase A / Phase B training curves  
- Test accuracy around **~99%** with the improved transfer-learning setup  
- Files under `models/*.keras`

---

## Risks & upgrades

1. **CPU-only Windows TF** — longer wall-clock; prefer GPU/WSL2 for full-dataset epochs.  
2. **Next metrics:** confusion matrix, precision/recall, calibration.  
3. **Optional:** stronger augmentation / mixup, longer fine-tune.

---

## Completion checklist

- [x] Dataset located / class folders prepared  
- [x] Visualization of dogs & cats  
- [x] Generators (`trdata` / `tsdata`)  
- [x] EfficientNet-B0 model compiled  
- [x] Callbacks + evaluation  
- [x] Model saved under `models/`  
- [x] Plan saved (`PLAN.md`)  
- [x] Run instructions documented  

# PlantVillage — Design & Decisions

## What This Project Does

Classify 54,306 images of healthy and diseased plant leaves (38 classes, 14 crop species) using a
ResNet-18 CNN trained from scratch, then iteratively optimized. Goal: beat or match Mohanty et al.
(2016) on macro-F1.

---

## Data Loading

The HF `datasets` library (v4.x) cannot run this dataset's custom loading script, so
`load_dataset("mohanty/PlantVillage")` returns only a text index of file paths — no images. Instead
we pull repo files directly via `huggingface_hub.hf_hub_download`:

- `data.zip` (~2.2 GB) — raw images under `raw/{color,grayscale,segmented}/<Class>/<file>`
- `splits/{cfg}_{train,test}.txt` — the official train/test partition per modality
- `leaf_grouping/leaf-map.json` — maps each image filename to its physical leaf

The `derive_leaf_id` function in the notebook replicates the exact logic from the dataset's
`plant_village.py` loading script so we recover the same physical-leaf grouping. The zip extracts
`raw/` directly under the cache root — no recursive search needed.

Images are decoded lazily by the PyTorch `Dataset` — none are loaded into RAM during EDA or
splitting.

---

## Splits

The dataset ships only train/test splits. We carve validation out of train using
`StratifiedGroupKFold` (group = `leaf_id`, stratify = `label`, ~15%, `SEED=42`):

- **Grouped by leaf**: the same physical leaf cannot appear in both train and val (mirrors how the
  official train/test split was built — a leaf cannot span both).
- **Stratified by label**: the class distribution is preserved in both train and val.
- The val leaf set is derived once from the color split and reused for grayscale and segmented,
  giving all three modalities an identical partition for fair comparison.

Results: color train/val/test ≈ 37,367 / 6,229 / 10,709. Val fraction varies slightly across
modalities (~11–14%) because per-modality filename sets differ, but all are leakage-free.

---

## Class Imbalance

PlantVillage is heavily skewed (37.7× largest vs smallest class):

| Largest | Count | Smallest | Count |
|---|---|---|---|
| Orange___Haunglongbing | 4,527 | Potato___healthy | 120 |
| Tomato___Yellow_Leaf_Curl_Virus | 4,257 | Raspberry___healthy | 209 |
| Soybean___healthy | 4,117 | Apple___Cedar_apple_rust | 223 |

Tomato alone is 33% of the training set (14,521 images).

**Approach — weight, don't resample:** `CrossEntropyLoss(weight=balanced_weights)` computed from
the train subset only (sklearn `compute_class_weight("balanced")`). Weights range from 0.25 (common
classes) to 9.46 (rare classes). Resampling was rejected because it either discards majority-class
data or duplicates 120-image classes into overfitting. Augmentation is the planned third lever in
the optimization phase.

---

## Model

Plug-and-play `torchvision.models.resnet18(weights=None)` with the final `fc` replaced by
`nn.Linear(512, 38)`. 11.2M parameters, trained from scratch. A one-line `PRETRAINED` toggle
switches to ImageNet transfer learning for the optimization phase.

---

## Training

- **Optimizer**: SGD with Nesterov momentum (`lr=0.05`, `momentum=0.9`, `weight_decay=1e-4`).
  Adam was rejected because its EMA update (`lerp_`) is not supported by the DirectML backend
  (AMD GPU on Windows) and falls back to CPU, negating the GPU.
- **Schedule**: CosineAnnealingLR over `EPOCHS` epochs.
- **Checkpointing**: best-by-val-macro-F1 weights saved to `outputs/resnet18_{config}_best.pt`.
- **GPU**: uses DirectML (`torch_directml`) for AMD GPU on Windows. `NUM_WORKERS=0` is required
  because DirectML + Windows multiprocessing (PyTorch `spawn`) causes workers to hang.
- **Batch size**: 128 (larger batches improve GPU utilization with DirectML).

---

## Augmentation (Baseline — Light)

Train: `RandomResizedCrop(224, scale=(0.7,1.0))` + `RandomHorizontalFlip`.  
Eval: `Resize(224)` + `CenterCrop(224)`.  
Both: ImageNet normalization (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).

Heavier augmentation (color jitter, rotation, etc.) is deferred to the optimization step.
Grayscale images are converted to 3-channel RGB (`.convert("RGB")`) so one ResNet-18 works
across all three modalities.

---

## Evaluation Metrics

- **Macro-F1** — primary metric; addresses class imbalance and rare disease recall
- **Accuracy** — secondary
- **Per-class precision / recall / F1** — via `classification_report`
- **Confusion matrix** — to analyze which diseases get confused
- Deferred to optimization phase: AUROC (one-vs-rest), Expected Calibration Error (ECE)

---

## Next Steps (Optimization Phase)

- Heavier augmentation: `ColorJitter`, `RandomRotation`, `RandomErasing`
- Transfer learning: set `PRETRAINED = True`, compare macro-F1 vs from-scratch baseline
- Full metric suite: per-class recall, AUROC, ECE
- Compare color vs grayscale vs segmented modalities
- Benchmark macro-F1 against Mohanty et al. (2016)

# Bone Fracture Classification
**Kamand Bioengineering Group Hackathon 2025 | IIT Mandi**

## Dataset & Reorganization
Before training, download and reorganize the 12-class dataset:

```bash
# Downloads "bone-break-classifier-dataset" via kagglehub and maps into train/val/test splits
python scripts/reorganize_multiclass.py
```
This script splits the **1,685** images into:
- 70% Train, 10% Validation, 20% Test (stratified across 12 fracture classes).

## Architecture
A **soft-voting ensemble** of three ImageNet-pretrained backbones:

| Backbone | Role | Weight |
|---|---|---|
| ViT-B/16 (`vit_base_patch16_224`) | Primary – global attention | 50% |
| EfficientNet-B3 | Compact CNN features | 25% |
| ConvNeXt-Small | Hierarchical local features | 25% |

Key design choices:
- All models fine-tuned from **ImageNet** (no fracture-specific pre-training, per rules)
- **Label smoothing** cross-entropy + **MixUp** augmentation for better generalization
- **Weighted class sampler** to handle imbalanced fracture categories
- **GradCAM** attention maps for radiologist-interpretable explainability
- 5-fold stratified cross-validation for reliable generalization estimates
- Apple Silicon (MPS) & CUDA auto-detection

## Dataset
Multi-class classification: **12 Fracture Types**
Avulsion, Comminuted, Compression-Crush, Fracture Dislocation, Greenstick, Hairline, Impacted, Intra-articular, Longitudinal, Oblique, Pathological, Spiral.
- Train: 1173 images | Val: 163 images | Test: 349 images
- Format: PNG/JPEG, variable resolution → standardized to 224×224

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install kagglehub

# 2. Run data download & reorganization
python scripts/reorganize_multiclass.py

# 4. Train (single run)
python train.py --config config.yaml

# 5. Train with 5-fold CV
python train.py --config config.yaml --cv

# 6. Evaluate on test set (generates all CSVs + plots)
python evaluate.py --config config.yaml
```

## Repository Structure
```
bone_fracture_classifier/
├── config.yaml                  # All hyperparameters
├── requirements.txt             # Pinned dependencies
├── data_loader.py               # Dataset, augmentation, CV splits
├── model.py                     # ViT / EfficientNet / ConvNeXt ensemble
├── train.py                     # Training loop + epoch CSV logger
├── evaluate.py                  # Test metrics, confusion matrix, GradCAM
├── scripts/
│   └── reorganize_multiclass.py # Automatic dataset download and stratified train/val/test splitting
├── TEAM.txt                     # Team information
├── README.md                    # This file
├── checkpoints/
│   └── best_model.pth           # Best checkpoint (val macro-F1)
└── results/
    ├── final_results.csv            # Submission metric sheet
    ├── model_performance_analysis.csv  # Epoch-by-epoch log
    ├── confusion_matrix.png
    ├── roc_curves.png
    └── gradcam_samples.png
```

## Augmentation Strategy
- Horizontal flip, rotation (±15°), affine transforms
- CLAHE contrast enhancement (X-ray specific)
- Grid distortion (simulates X-ray artifacts)
- MixUp (α=0.2) for regularization
- Coarse dropout (random erasing)
- ImageNet normalization

## Hyperparameters
| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size (effective) | 64 (16 × 4 accumulation) |
| Optimizer | AdamW (lr=5e-5, wd=1e-2) |
| Scheduler | Cosine + 5-epoch warmup |
| Loss | Label Smoothing CE (ε=0.1) |
| Epochs | 40 (early stop patience=10) |
| Mixed precision | fp16 (CUDA only) |

## Contact
Team members listed in TEAM.txt

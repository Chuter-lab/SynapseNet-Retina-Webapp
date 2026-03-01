# Supplementary Methods

## S1. Training Configuration Details

### Hardware
- GPU: NVIDIA TITAN V (12 GB VRAM)
- CPU: [PLACEHOLDER: CPU model]
- Server: eyenet-node01, UTHSC HPC

### Software Environment
- Python 3.10
- PyTorch 2.6.0
- SynapseNet 0.4.1
- torch_em 0.7.8
- micro-SAM 0.4.1
- CUDA [PLACEHOLDER: version]

### Data Preprocessing
- Raw montages: 7000 x 7000 px, 8-bit grayscale TIF
- Annotated crops: 1825 x 1410 px, extracted at ribbon synapse locations
- Labels: Instance segmentation masks (integer-valued, 0=background)
- Train/val/test split: Sections 0,1 / Section 1 / Section 2

### Unannotated Data for Domain Adaptation
- 12 montage images (4 stacks x 3 sections)
- Converted to HDF5 format with 'raw' dataset key
- Used for mean-teacher semi-supervised training

## S2. Hyperparameter Summary

| Parameter | Fine-tune (focal) | Fine-tune (dice) | Domain Adapt | DA+FT | micro-SAM FT |
|-----------|-------------------|-------------------|--------------|-------|-------------|
| Learning rate | 1e-4 | 1e-5 | 1e-4 | 1e-4 | [PLACEHOLDER] |
| Iterations | 5000 | 5000 | 5000 | 5000 | [PLACEHOLDER] |
| Batch size | 8 | 2 | 4 | 8 | 1 |
| Patch shape | 256x256 | 512x512 | 256x256 | 256x256 | 512x512 |
| Loss | FocalDice | Dice | MSE (teacher) | FocalDice | SAM default |
| Optimizer | Adam | Adam | Adam | Adam | AdamW |
| Scheduler | ReduceLR | ReduceLR | — | ReduceLR | — |
| Augmentation | Yes | No | — | Yes | — |
| Out channels | 1 | 2 | 2 | 1 | SAM |
| Final activation | None | Sigmoid | Sigmoid | None | — |

### Focal Loss Configuration
- Focal gamma: 2.0
- Foreground weights: vesicles=20.0, mitochondria=5.0, membrane=2.0
- Combined focal + Dice loss

### Learning Rate Schedule
- ReduceLROnPlateau: mode=min, factor=0.5, patience=10, min_lr=1e-7

### Data Augmentation (fine-tuning only)
- Random horizontal/vertical flips
- Random rotations (90°)
- Elastic deformation
- Gaussian noise
- Gaussian blur

## S3. Post-processing Parameters

| Structure | Min Instance Area (px) | Morphological Opening Radius |
|-----------|----------------------|------------------------------|
| Vesicles | 50 | 2 |
| Mitochondria | 5000 | 2 |
| Membrane | 50000 | 2 |

Threshold: 0.5 (binary segmentation from sigmoid/logit output)

## S4. Weight Transfer Protocol

When adapting pretrained SynapseNet models (out_channels=2) for focal loss fine-tuning (out_channels=1):
1. Initialize new UNet2d with out_channels=1 and no final activation
2. Load pretrained/DA state dict
3. Transfer all matching weights (same key name and tensor shape)
4. Result: 44/46 tensors transferred; out_conv.weight (2,32,1,1 → 1,32,1,1) and out_conv.bias (2 → 1) re-initialized with default PyTorch initialization

## S5. Channel Convention
SynapseNet pretrained models output 2 channels:
- Channel 0: Foreground probability
- Channel 1: Boundary probability

For fine-tuned models (out_channels=1), only foreground is predicted.

## S6. Combination Optimization Protocol

[PLACEHOLDER: Describe threshold sweep (0.1-0.9 in 0.05 steps), TTA with 7 augmentations (identity, flip-H, flip-V, flip-HV, rot90, rot180, rot270), and ensemble strategies (average, maximum, majority voting). Include final optimized results.]

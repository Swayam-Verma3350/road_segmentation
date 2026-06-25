# Experimental Report: Road Segmentation from Satellite Imagery

**Task:** Binary semantic segmentation (road vs. background) on aerial imagery  
**Dataset:** Massachusetts Roads Dataset (Mnih, 2013)  
**Framework:** PyTorch 2.x, Albumentations, segmentation-models-pytorch  
**Compute:** Google Colab (T4 GPU, mixed precision)

---

## 1. Problem Setup

Road extraction from satellite imagery is a binary semantic segmentation task: given an RGB aerial tile, predict a per-pixel binary mask indicating road presence. The task is challenging for several structural reasons:

**Class imbalance.** Road pixels constitute roughly 3–5% of each tile, giving a background-to-road pixel ratio of approximately 20–35:1 in the training split. A naive model can achieve ~95% pixel accuracy by predicting all-background — a degenerate solution that must be explicitly guarded against in loss design.

**Thin, connected structures.** Roads are spatially thin (2–6 px wide at the training resolution of 256×256) and form long, connected networks. Standard per-pixel losses do not directly penalize breaks in connectivity, which remain a systematic failure mode across all models tested.

**Label noise.** Masks are rasterized from OpenStreetMap road centerlines with a fixed line thickness, not hand-traced from the imagery. This introduces two sources of noise: (1) registration offset between the rasterized mask and the visible road center, typically a few pixels, especially on curves; (2) OSM coverage gaps that appear as missing labels rather than absent roads. Both effects impose a ceiling on achievable IoU that is independent of model quality.

**Scale variation.** Road width varies considerably across scene types — from wide multi-lane highways to single-track rural roads — within a single dataset. Architectures with multi-scale receptive fields are theoretically better suited to handle this variation.

---

## 2. Dataset & Preprocessing

### Dataset Statistics

| Split | Tiles | Notes |
|---|---|---|
| Train | 1,108 | 25% subset (≈277 tiles) used for all experiments |
| Val | 14 | Always evaluated in full |
| Test | 49 | Always evaluated in full |

All tiles are 1500×1500 px at approximately 1 m/px resolution. The official train/val/test split is respected without reshuffling to avoid geographic leakage between splits.

A 25% training subset was chosen due to Colab compute constraints — training five separate models on 1108 full-resolution tiles within a single session is intractable. The same random subset (fixed seed=42) is used for every experiment, ensuring all models are trained on identical data.

### Road Width Distribution

Using a distance-transform approach to estimate per-pixel road width from the training masks (width ≈ 2× distance to nearest non-road pixel), the road width distribution is:

- Mean: approximately 3–5 px at 256×256 resolution
- P90: approximately 7–9 px
- Dominant at 1–4 px: the majority of road pixels belong to thin, narrow structures

At this resolution thin roads often span only 1–2 pixels, making high-precision segmentation geometrically difficult.

### Augmentation Pipeline (training only)

| Transform | Parameters | Rationale |
|---|---|---|
| Resize | 256×256 | Uniform input size |
| HorizontalFlip | p=0.5 | Aerial view is orientation-free |
| VerticalFlip | p=0.5 | Same |
| RandomRotate90 | p=0.5 | 90° rotational symmetry |
| Affine | scale (0.9–1.1), translate ≤5%, rotate ±15° | Small spatial perturbations |
| RandomBrightnessContrast | ±20% | Lighting variation |
| GaussNoise | p=0.2 | Sensor/compression noise |
| Normalize | ImageNet mean/std | Required for pretrained encoder |

Validation and test sets use only resize + normalize, applied identically to all five models.

---

## 3. Models

### 3.1 U-Net (Baseline)

A standard 4-level encoder-decoder with skip connections, implemented from scratch. The encoder uses `ConvBlock` units (two 3×3 Conv → BatchNorm → ReLU), with MaxPool2d downsampling between levels. The decoder uses bilinear upsampling followed by concatenation with the corresponding encoder skip feature map, then another ConvBlock.

Channel widths: 32 → 64 → 128 → 256 (encoder), mirrored in the decoder.

Total parameters: approximately 7.8M.

This model serves as the reference for all comparisons: architectural experiments hold the loss fixed at BCE (same as the baseline), and loss experiments hold the architecture fixed at this U-Net.

### 3.2 Attention U-Net

Identical to the U-Net baseline except that each skip connection passes through an **additive attention gate** (Oktay et al., 2018) before concatenation with decoder features.

The gate takes two inputs: the decoder feature map at the current scale (the "gating signal", g) and the encoder skip features (x). It projects both to a shared intermediate space, adds them, applies a learned sigmoid mask, and multiplies the mask element-wise onto x. This allows the model to learn which spatial regions of the encoder features are relevant for each decoding step, suppressing background-heavy activations.

**Motivation for this choice:** given the 20–35:1 class imbalance, the plain U-Net skip connections pass large quantities of background-dominated activations directly into the decoder. Attention gates directly target this failure mode with minimal added computation (a few small 1×1 convolutions per gate). This was the most principled first architectural extension to try before adding pretrained encoders or changing the overall architecture family.

### 3.3 DeepLabV3+ (ResNet34 encoder)

A DeepLabV3+ model from `segmentation-models-pytorch`, using a pretrained ResNet34 encoder and an **ASPP (Atrous Spatial Pyramid Pooling)** module.

ASPP applies parallel dilated convolutions at dilation rates {1, 6, 12, 18} alongside a global average pooling branch, then concatenates all outputs. This gives the model simultaneous access to local fine-grained detail and large-context information without increasing the number of parameters proportionally.

**Motivation for this choice:** roads are contextual structures — a road pixel is easier to detect if the model can see that a line continues in a consistent direction across a large spatial extent. ASPP is specifically designed to increase effective receptive field for exactly this use case. The pretrained ResNet34 encoder additionally gives the model useful low-level edge detectors and texture features from ImageNet, reducing the effective sample complexity on our small training subset.

Total parameters: approximately 15.4M (more than double the U-Net variants).

---

## 4. Loss Functions

All experiments use **class-weighted BCE** as the base loss. The `pos_weight` value is derived empirically from the measured road pixel percentage in a 150-image training sample:

```
pos_weight = background:road ratio = (100 / mean_road_pct) - 1  [clipped to range 1–10]
```

This prevents the trivial shortcut of predicting all-background, which BCE without pos_weight permits.

### 4.1 BCE (baseline loss)

Standard binary cross-entropy with `pos_weight`. Applied to every pixel independently.

### 4.2 BCE + Dice (50/50 blend)

```
L = 0.5 · BCE(logits, targets) + 0.5 · DiceLoss(logits, targets)
```

Dice loss is computed as `1 - (2·|pred ∩ target| + ε) / (|pred| + |target| + ε)` over the full image, making it invariant to class frequency. Blending with BCE retains stable per-pixel gradients (important for early training convergence) while adding a direct overlap objective that is not confused by class imbalance.

### 4.3 Focal Loss

```
FL(p_t) = -alpha_t · (1 - p_t)^gamma · log(p_t)
```

with γ=2 and class-weighted `alpha_t` derived from the same measured imbalance as `pos_weight`:

```
alpha_t = pos_weight_value / (pos_weight_value + 1)  [≈ class prior for road class]
```

The `(1 - p_t)^gamma` factor down-weights easy, confidently-correct background predictions, concentrating gradient on hard misclassified pixels. The class-weighted `alpha_t` ensures the road class receives proportionally higher loss regardless of the modulation. A flat scalar alpha (a common implementation error) would not provide this class-rebalancing effect.

---

## 5. Training Protocol

All five models share the same training loop:

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 5×10⁻⁴ |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2, monitor: val IoU) |
| Max epochs | 25 |
| Early stopping patience | 5 epochs (monitor: val IoU) |
| Gradient clipping | Global norm ≤ 1.0 |
| Mixed precision | torch.amp (CUDA only) |
| Batch size | 8 |
| Checkpoint criterion | Best val IoU |

Best checkpoints are restored before test evaluation, so test metrics reflect peak validation performance rather than final epoch state.

---

## 6. Results

### 6.1 Test Set Quantitative Results

| Model | IoU | Dice | Precision | Recall | F1 | Pixel Acc. |
|---|---|---|---|---|---|---|
| **U-Net (BCE+Dice)** | **0.2662** | **0.4165** | 0.2906 | **0.7685** | **0.4165** | **0.9094** |
| Attention U-Net (BCE) | 0.2590 | 0.4077 | 0.2881 | 0.7341 | 0.4077 | 0.9080 |
| U-Net baseline (BCE) | 0.2565 | 0.4047 | 0.2822 | 0.7489 | 0.4047 | 0.9052 |
| U-Net (Focal Loss) | 0.1743 | 0.2921 | 0.2084 | 0.6019 | 0.2921 | 0.8732 |
| DeepLabV3+ (BCE) | 0.1657 | 0.2832 | 0.2025 | 0.5445 | 0.2832 | 0.8802 |

### 6.2 Architecture Comparison (BCE held fixed)

The Attention U-Net achieved a best val IoU of 0.2613 versus the baseline's 0.2557 — a modest improvement of +2.2% relative. Both models show broadly similar training dynamics: smooth val loss decline with gradually improving IoU throughout training (no early saturation). The attention mechanism appears to provide consistent but small gains, suggesting that on this dataset the encoder feature selection problem is real but not dominant.

DeepLabV3+ underperformed both U-Net variants substantially (test IoU = 0.1657), despite having the largest parameter count and a pretrained encoder. The most likely explanation is that the ResNet34 encoder's large effective receptive field and strided downsampling at the early layers loses the spatial precision needed to reconstruct thin 1–2 px roads at 256×256 resolution. U-Net's skip connections recover fine spatial detail at each scale; the DeepLabV3+ decoder, while effective at semantic segmentation of larger objects, may be insufficient for this spatially fine-grained task without structural modifications.

### 6.3 Loss Function Comparison (U-Net held fixed)

BCE+Dice achieved the highest overall test IoU (0.2662) — a +3.8% relative improvement over the BCE baseline. More notably, it also achieved the highest recall (0.7685), consistent with Dice loss's property of directly optimizing overlap and penalizing false negatives symmetrically with false positives.

Focal Loss underperformed both BCE variants (IoU = 0.1743), which is a surprising result. Possible explanations:

1. The `(1 - p_t)^gamma` modulation may over-suppress early-training gradients on a dataset where the training split is small (277 tiles), slowing convergence relative to the 25-epoch budget.
2. The gamma=2 hyperparameter was not tuned — a lower gamma might preserve more gradient signal during early training while still emphasizing hard examples later.
3. Focal Loss is most effective when the dominant problem is easy negatives overwhelming gradient, but in this case the class-weighted BCE with `pos_weight` already addresses the core imbalance problem — Focal Loss may be targeting a problem that was already handled.

### 6.4 Training Dynamics

**U-Net baseline (BCE):** val loss started high (~3.3) due to the large initial BCE with high pos_weight, then converged rapidly within 2–3 epochs. Val IoU increased continuously throughout training without signs of overfitting, suggesting the model was still learning at epoch 25. IoU reached ~0.23–0.24 by the end of training.

**Attention U-Net (BCE):** smoother initial training than the baseline (val loss starting at ~1.0, no spike), with a dip around epochs 4–5 that recovered. This suggests the attention gates need a few epochs to initialize usefully. Final IoU trajectory slightly above the baseline.

**DeepLabV3+ (BCE):** showed a large initial val loss spike (~2.0) followed by rapid convergence to a plateau around val loss 0.8. Val IoU plateaued early (~epoch 5) at 0.15–0.18 and did not improve substantially, despite 15 epochs of additional training. This plateau behavior distinguishes it from the U-Net variants, which continued improving.

**U-Net (BCE+Dice):** the blended loss resulted in tighter train/val loss tracking throughout training — both curves declined together without the large train/val gap seen in BCE-only models. Val IoU improved steadily and was the highest of all models at most epochs after epoch 10.

**U-Net (Focal Loss):** loss values are on a different scale (0.016–0.023) and therefore not directly comparable across the validation loss comparison plot. Val IoU peaked around epoch 11–12 (~0.20) and declined slightly thereafter, suggesting mild overfitting relative to the other models.

### 6.5 Qualitative Analysis

Visual inspection of predictions on test images reveals consistent patterns across all models:

- All U-Net variants successfully identify major roads and highway junctions, capturing the broad connectivity of the road network.
- DeepLabV3+ produces notably blurrier, more dilated predictions, often merging adjacent road segments into large connected blobs. This is consistent with the architecture's larger receptive field causing spatially imprecise boundaries.
- BCE+Dice and Attention U-Net produce the sharpest, most structurally clean predictions on well-represented road types.
- All models struggle with: (1) narrow rural tracks that are 1–2 px wide at training resolution, (2) roads partially occluded by tree canopy, and (3) regions where OSM coverage gaps create missing labels that appear as model errors.

---

## 7. Summary of Findings

**On architecture:** Attention U-Net provides a small but consistent improvement over the plain U-Net baseline when the loss function is held constant. The attention mechanism helps suppress background features on the class-imbalanced skip connections. DeepLabV3+ underperformed despite its advantages on the board — its spatial precision limitations outweigh its multi-scale context advantage for thin-road segmentation at 256×256 resolution.

**On loss functions:** BCE+Dice is the best-performing loss configuration on this dataset. The combination of a pixel-level loss for stable gradients and an overlap-based loss for class-balance-invariant supervision outperforms both pure BCE and Focal Loss. Focal Loss did not provide the expected benefit, likely due to interaction with the small training set size and the 25-epoch budget.

**Across all experiments:** the gains from both architectural improvements and loss modifications are modest in absolute terms (best IoU 0.2662 vs. baseline 0.2565). The primary performance bottleneck is not the choice of architecture or loss but rather the combination of small training set size, severe downsampling from 1500px to 256px, and the inherent label noise in OSM-rasterized annotations. Scaling training data and resolution would likely yield larger gains than further model or loss modifications at this scale.

---

## 8. Reproducibility Notes

- All experiments use `SEED=42` for Python `random`, NumPy, and PyTorch.
- The same 25% training subset (chosen with fixed seed) is used across all five models.
- `torch.backends.cudnn.deterministic` is not forced (for performance), so minor run-to-run variation on GPU is possible.
- Best checkpoints are saved by val IoU and restored before test evaluation.

---

## References

1. Mnih, V. (2013). *Machine Learning for Aerial Image Labeling*. PhD thesis, University of Toronto.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*. https://arxiv.org/abs/1505.04597
3. Oktay, O. et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. *MIDL*. https://arxiv.org/abs/1804.03999
4. Chen, L.-C. et al. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. *ECCV*. https://arxiv.org/abs/1802.02611
5. Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV*. https://arxiv.org/abs/1708.02002
6. Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. https://arxiv.org/abs/1606.04797

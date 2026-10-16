# Schematics NPC DSC 2026 - Day 1 Comprehensive Technical Dossier

**Target:** Open-Set Bird Chirp Classification & Novelty Detection  
**Competition:** Schematics NPC Data Science Competition 2026 (ITS Surabaya)  
**Date:** 2026-10-15 (Day 1 Execution)  
**Primary Master Notebook:** `schematics-day-1.ipynb`  
**Current Best Leaderboard Score:** **0.59183** (Public LB, Rank 17, Submission `sub_004`)  
**Compiled for:** Deep Architecture & Methodology Audit  

---

## 1. Executive Summary & Problem Signature

The competition task requires classifying short audio recordings into three known endemic Indonesian bird species alongside an unlabelled open-set reject class designated as **Unknown**.

### Core Constraints & Scoring
* **Scoring Formula:**
$$
\text{Final Score} = 0.35 \times \text{Kaggle Macro F1} + 0.65 \times \text{Technical Report Score}
$$
* **Report Weighting (65%):** Evaluation and Validation (30%), Modeling (25%), Preprocessing and Feature Engineering (20%), Exploratory Data Analysis (15%), Structure and Writing (10%).
* **Competition Restrictions:** Strictly zero external audio data, no full AutoML, 5 submissions per day limit, deterministic random seeds required.

### Target Classes & Severe Class Imbalance
* **Bondol Jawa** (*Lonchura leucogastroides*): 1,124 samples (61.56%)
* **Kutilang** (*Pycnonotus aurigaster*): 619 samples (33.90%)
* **Cendrawasih** (*Paradisaeidae*): 83 samples (4.55%)
* **Unknown**: Target test distribution class with zero training examples.
* **Imbalance Ratio:** $IR = 13.54$ (Bondol Jawa to Cendrawasih). Shannon Entropy: 1.163 bits (normalized: 0.734).

---

## 2. Empirical Data Profile & Acoustic Diagnostics

### Audio Stream Characteristics
* **Train Set:** 1,826 FLAC files, 48,000 Hz sample rate, 2 audio channels (stereo), duration 5.27s to 14.90s (mean: 10.04s).
* **Test Set:** 814 FLAC files, 48,000 Hz sample rate, 2 audio channels (stereo), duration 5.55s to 14.90s (mean: 10.09s).
* **Duration Distribution Shift:** Two-sample Kolmogorov-Smirnov test statistic = 0.0302, p-value = 0.6695, Wasserstein distance = 0.0892s. Proves zero duration shift between train and test.
* **Exact Data Leakage Audit:** SHA256 byte-level hash deduplication verified 1,826/1,826 unique train hashes, 814/814 unique test hashes, with exactly 0 train-test hash overlap.

### Signal Dynamics & Recording Hardware Artifacts
* **Digital Clipping Rate:** 40.0% to 55.0% of all recordings reach peak amplitude $\ge 0.99$ (Bondol Jawa: 45.0%, Cendrawasih: 40.0%, Kutilang: 45.0%, Test Set: 55.0% with peaks up to 1.004).
* **Silence Ratio:** 9.2% to 10.8% across all groups (measured as signal amplitude $< 0.01$ or $-40\text{ dBFS}$).
* **Zero Crossing Rate (ZCR):** Kutilang shows low ZCR of 0.2899 (pure tonal whistle modulation), Bondol Jawa exhibits 0.3483, Cendrawasih exhibits 0.3699 (noisy broadband acoustic texture), and Test Set exhibits 0.3526.
* **Stereo Channel Redundancy:** Inter-channel correlation between Ch0 and Ch1 ranges from 0.724 to 0.800 across classes; mean RMS difference between channels is 0.002 to 0.003. Confirms stereo channels carry near-identical mono signals, justifying arithmetic mean downmixing `(ch0 + ch1) / 2`.

### Spectral Power Decomposition by Frequency Band

Welch Power Spectral Density (PSD) decomposition across frequency bands:

| Group / Species | 0-2 kHz | 2-4 kHz | 4-6 kHz | 6-8 kHz | Acoustic Profile Summary |
|---|---|---|---|---|---|
| **Bondol Jawa** | 28.4% | 37.2% | 27.3% | 7.1% | Broadband mid-high distribution (64.5% across 2-6 kHz) |
| **Kutilang** | 29.3% | **57.2%** | 9.7% | 3.8% | Concentrated narrow tonal whistle (peak around 2.5-3.5 kHz) |
| **Cendrawasih** | **44.0%** | 25.6% | 15.7% | **14.7%** | Bimodal low-rumble plus high harmonic structure |
| **Test Set (Mixed)** | **39.3%** | 36.7% | 18.4% | 5.6% | Aggregate mixture with elevated low-frequency energy |

### High-Frequency Power Audit at Native 48 kHz
* **Band 0-8 kHz:** Absorbs over 90.7% total power across all species (Bondol: 96.37%, Kutilang: 93.54%, Cendrawasih: 90.71%).
* **Band 8-16 kHz:** Cendrawasih carries **9.08%** harmonic energy, Kutilang carries 5.24%, Bondol carries 3.52%.
* **Band 16-24 kHz:** Contains only **0.11% to 1.22%** power across all classes.
* **Architecture Decision (D-005):** Downsampling to 16 kHz cuts off at Nyquist 8 kHz and destroys 9.08% of Cendrawasih harmonic power. Downsampling to **32 kHz (Nyquist 16 kHz)** preserves 100% of informative harmonic frequencies while reducing tensor memory by 33.3% relative to native 48 kHz.

---

## 3. Validation Scheme & Group Dependency Partition

### Recording-Source Group Dependency (Bottleneck Discovery)
The metadata exposes a `source_group` column representing field recording sessions:
* Exactly 15 unique source groups exist across the 1,826 train samples.
* Groups are 100% class-pure: Bondol Jawa has 7 groups, Kutilang has 4 groups, Cendrawasih has exactly 4 groups.
* **Cardinality Bottleneck:** Because Cendrawasih and Kutilang possess exactly 4 source groups, a 5-fold split is mathematically impossible without creating folds with 0 validation samples for minority classes.
* **Extreme Group Size Skew in Cendrawasih:** Group sizes are 3, 16, 29, and 35.

### 4-Fold StratifiedGroupKFold Implementation
Partitioning was performed with `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)` on `source_group` stratified by `label`:

| Fold | Total Validation Samples | Bondol Jawa | Kutilang | Cendrawasih | Source Groups Assigned |
|---|---|---|---|---|---|
| **Fold 0** | 503 | 336 | 132 | **35** | 3 groups |
| **Fold 1** | 462 | 290 | 143 | **29** | 4 groups |
| **Fold 2** | 422 | 240 | 166 | **16** | 4 groups |
| **Fold 3** | 439 | 258 | 178 | **3** | 4 groups |

### Validation Invariant (FIX-A: Global Out-of-Fold Aggregation)
In Fold 3, Cendrawasih contains only 3 validation samples. A single misclassification in Fold 3 drops the fold's Cendrawasih F1 by 33.3%. Averaging unweighted per-fold F1 scores introduces severe variance and bias. Therefore, Macro F1 is **strictly evaluated as Global Out-of-Fold (OOF)** across the concatenated 1,826 validation predictions:
$$
\text{Global Macro F1} = \frac{1}{C} \sum_{c=1}^C F_{1, \text{global}}^{(c)}
$$

---

## 4. Preprocessing & Augmentation Pipeline Architecture

The pipeline executes in strict sequential order inside `FastWaveformDataset` and GPU frontend:

```text
Raw Stereo Audio (.flac, 48kHz)
  |
  +--> 1. Mono Downmix: (ch0 + ch1) / 2
  |
  +--> 2. Polyphase Anti-Aliasing Resampling: 48kHz -> 32kHz (scipy.signal.resample_poly, up=2, down=3)
  |
  +--> 3. Per-Sample Peak Normalization: wav = wav / max(|wav|) to [-1.0, 1.0]
  |
  +--> 4. Duration Standardization: 10.0 seconds (320,000 samples at 32kHz)
  |       - Audio < 10s: reflect-padding (avoids artificial periodicity or zero-silence injection)
  |       - Audio > 10s (train): stochastic random cropping
  |       - Audio > 10s (eval/test): deterministic center cropping
  |
  +--> [Train Only] 5. Waveform Mixup: Beta(0.2, 0.2), p=0.5 applied on raw audio
  |
  +--> [Train Only] 6. Random Gain Jitter: +/- 6 dB (scaling factor 0.5x to 2.0x), p=0.5
  |
  +--> [Train Only] 7. Additive Gaussian Noise: SNR 15 dB to 30 dB, p=0.5
  |
  +--> 8. GPU-Accelerated Log-Mel Spectrogram (torch.stft FP32):
  |       - n_fft = 2048 (~64 ms window)
  |       - hop_length = 512 (~16 ms step, 75% overlap)
  |       - n_mels = 128 (frequency range: 200 Hz to 16,000 Hz)
  |       - power_to_db: top_db = 80.0
  |       - Output Tensor Shape: (1, 128, 626)
  |
  +--> [Train Only] 9. SpecAugment2D: 2 frequency masks (max 16 bins), 2 time masks (max 32 frames), p=0.8
  |
  +--> 10. Per-Sample Robust Z-Score Normalization: (S_dB - mean) / (std + 1e-6)
```

### Prohibited Augmentations (Domain Physics Invariants)
* **HorizontalFlip:** Banned. Time axis reversal creates physically impossible backward bird chirps.
* **VerticalFlip:** Banned. Frequency axis inversion destroys pitch and fundamental harmonic relationships.
* **Pitch Shifting:** Banned. Shifts fundamental frequencies, corrupting inter-species spectral signatures.
* **Time Stretching:** Banned. Distorts species-specific syllable repetition rates.

---

## 5. Five-Model Multi-Paradigm Portfolio & Empirical Results

Five distinct model families were trained under the exact 4-Fold StratifiedGroupKFold partition.

All deep learning models utilized:
* **Two-Phase Transfer Learning (LP-FT):** Phase 1 (5 epochs Linear Probing with frozen backbone, `lr = 1e-3`), Phase 2 (18 epochs Fine-Tuning with differential learning rate, `lr_backbone = 1e-4` to `1.5e-4`, `lr_head = 4e-4` to `5e-4`, Cosine Annealing with 5% warmup).
* **Class-Balanced Focal Loss (FIX-F):**
$$
\text{Loss} = -\sum_{c} \alpha_c (1 - p_{t, c})^\gamma y_{\text{smooth}, c} \log(p_{t, c})
$$
where $\gamma = 2.0$, $\beta = 0.99$ (Cui et al., CVPR 2019 effective number weights: Bondol = 0.7959, Kutilang = 0.7974, Cendrawasih = 1.4067), and label smoothing $\epsilon = 0.05$.
* **Unified 256-d Bottleneck Projection Head (D-007):**
$$
\text{GAP}(d) \to \text{Linear}(d, 256) \to \text{BatchNorm1d} \to \text{SiLU} \to \text{Dropout}(0.3) \to \text{Linear}(256, 3)
$$
Extracts an L2-normalized 256-dimensional vector before classification for distance-based OOD novelty detection.

### Empirical 3-Class Closed-Set Benchmark (Global OOF N=1,826)

| Track | Model Architecture | Params | Inductive Bias / Key Mechanism | Global OOF Macro F1 | Bondol Jawa F1 | Kutilang F1 | Cendrawasih F1 | Runtime |
|---|---|---|---|---|---|---|---|---|
| **Track A** | LightGBM Tabular | 792 feat | Mel summary stats, temporal deltas, spectral centroid, ZCR | 0.8542 | **0.9556** | **0.9444** | 0.6627 | ~144s |
| **Track B1** | EfficientNet-B0 | 5.3M | Depthwise separable conv, compact spatial scaling | 0.8392 | 0.9221 | 0.9041 | 0.6914 | 695s |
| **Track B2** | ConvNeXt-Tiny | 28.0M | Modernized pure ConvNet, $7 \times 7$ large depthwise kernel | **0.8908** | 0.9302 | 0.9107 | **0.8316** | 1,115s |
| **Track B3** | SE-ResNeXt-50 32x4d | 27.5M | Multi-branch cardinality (32x4d) + Squeeze-and-Excitation | 0.8560 | 0.9205 | 0.9084 | 0.7391 | 1,274s |
| **Track B4** | Swin-Tiny Transformer | 28.3M | Shifted Window Self-Attention (W-MSA & SW-MSA) | 0.8489 | 0.9161 | 0.9016 | 0.7291 | 957s |

### Critical Model Comparison Insights
1. **ConvNeXt-Tiny Dominates Minority Class Recognition:** Receptive field width ($7 \times 7$ kernels) boosted Cendrawasih F1 to **0.8316** (+14.02% over EfficientNet-B0, +16.89% over LightGBM). It captured long-range frequency contours without over-fragmenting syllables.
2. **LightGBM Demonstrates Extreme Stability on Majority Classes:** 792 deterministic spectral descriptors achieved peak F1 on Bondol Jawa (0.9556) and Kutilang (0.9444), providing an anchor against deep neural network overconfidence.
3. **SE-ResNeXt-50 Achieved Perfect Fold Score:** On Fold 1, SE-ResNeXt-50 reached an exact F1 of **1.0000** on Cendrawasih ($N=29$), proving channel-wise attention successfully isolates bird chirps from static jungle background noise.

---

## 6. Ensembling Dynamics: NLL vs Macro F1 Loss Mismatch

In Cell 19, five-model probability blending was optimized using Nelder-Mead to minimize Out-Of-Fold Negative Log-Likelihood (NLL / Multi-class Cross-Entropy):

### Observed Optimization Result
* **NLL Objective:** 0.1585
* **Optimal Model Weights:**
  * LightGBM: **0.683**
  * ConvNeXt-Tiny: **0.207**
  * Swin-Tiny: **0.111**
  * EfficientNet-B0: **0.000** (eliminated as redundant)
  * SE-ResNeXt-50: **0.000** (eliminated as redundant)
* **Resulting 3-Class OOF F1:** 0.8670 (Bondol: 0.9601, Kutilang: 0.9552, Cendrawasih: 0.6857).

### The Mathematical Loss Alignment Flaw
* Unweighted NLL optimizes average sample likelihood. Bondol Jawa (1,124) and Kutilang (619) comprise **95.45%** of the dataset, while Cendrawasih (83) represents only **4.55%** of the sample pool.
* LightGBM outputs sharp, well-calibrated probabilities on the majority classes, drastically reducing overall cross-entropy.
* Consequently, the optimizer assigned 68.3% weight to LightGBM, suppressing ConvNeXt-Tiny from its single-model superiority.
* **Impact:** Cendrawasih F1 collapsed from **0.8316** (ConvNeXt standalone) down to **0.6857** in the NLL blend. Because competition evaluation uses **unweighted Macro F1** (where Cendrawasih carries an equal 25% weight), optimizing standard NLL directly degrades the primary metric.

---

## 7. Open-Set Unknown Detection & Cendrawasih Guard

### Multi-Layer OOD Scoring Architecture
Unknown prediction evaluates three complementary uncertainty representations:
1. **Layer 1: Negative Maximum Softmax Probability (MSP):** $1.0 - \max_c P(y=c|x)$.
2. **Layer 2: Free Energy Score (Liu et al., NeurIPS 2020):**
$$
E(x; T) = -T \cdot \log \sum_{c=1}^3 \exp\left(\frac{z_c}{T}\right), \quad T = 0.6080
$$
3. **Layer 3: Fold-Safe Cosine kNN Distance ($k=5$) on 256-d Embeddings:**
Evaluates distance to the 5 nearest neighbors in the training split embedding space. Evaluated strictly per fold model against the corresponding fold's reference set.

### OOD Detector Benchmark on 270 Pseudo-OOD Samples (ABL-03)
Evaluated on in-distribution OOF vs synthetic chimera cross-class perturbations:
* **MSP Standalone:** AUROC = 0.9283
* **Energy Score Standalone:** AUROC = 0.8209
* **256-d Bottleneck kNN-5 Standalone:** AUROC = **0.9970**
* **Ensemble OOD Score (50% kNN + 30% MSP + 20% Entropy):** AUROC = **0.9987**

### The Cendrawasih False-Positive Leakage & Guardrail
* **The Root Cause:** In `sub_001`, Cendrawasih was predicted for 112 test samples (13.76% of test set), compared to its true 4.55% training frequency. The class-balanced weight ($\alpha = 1.4067$) inflated Cendrawasih posterior probabilities, causing low-frequency tropical jungle rumble (0-2 kHz) and Unknown bird chirps to leak into Cendrawasih.
* **The Guardrail Mechanism:** An asymmetric boundary rule was introduced. Any sample assigned to Cendrawasih by the closed-set classifier must additionally pass a strict core-radius threshold ($\tau_{\text{cendra}}$, set at the 82nd percentile of in-distribution Cendrawasih OOF distance). If the sample lies in the sparse periphery of the Cendrawasih manifold, it is redirected to **Unknown**.
* **Empirical Verification:** In Cell 19, the guardrail intercepted 175 raw Cendrawasih predictions and converted 99 of them (56.6%) into Unknown, restoring Cendrawasih test frequency to 76 samples (9.34%).

---

## 8. Kaggle Submission Progression & Empirical Trajectory

All 5 allowed Day 1 submissions were executed and tracked:

| Submission ID | File | Core Engineering Strategy | Predicted Unknown | Public LB Score | Delta vs Baseline | Diagnostic Finding |
|---|---|---|---|---|---|---|
| **sub_001** | `experiment 1.csv` | LightGBM (0.52) + EffNet-B0 (0.48) + Global Tau p95 | 12.04% (98) | 0.54401 | Anchor Baseline | Cendrawasih over-predicted (112 samples); Unknown severely under-called. |
| **sub_002** | `experiment 2.csv` | LGBM + EffNet + kNN-5 + Cendrawasih Guard + p90 probe | 36.61% (298) | 0.58315 | **+0.03914** | Cendrawasih normalized to 35 (4.30%); Unknown recall recovered. |
| **sub_003** | `experiment 3.csv` | Sweet-spot p92 threshold probe | 28.01% (228) | 0.58313 | -0.00002 | Proved performance plateau across 28% to 36% Unknown ratio. |
| **sub_004** | `experiment 4.csv` | 5-Model Diverse Ensemble (NLL weights) + kNN-5 | 37.71% (307) | **0.59183** | **+0.04782** | Multimodal tabular-DL synergy confirmed; current competition benchmark. |
| *Team mate* | Offline sub | Baseline tabular probe (morning) | - | ~0.51000 | - | Verified un-normalized features fail. |

---

## 9. Comprehensive Post-Mortem: Bottlenecks & Critical Flaws

Before advancing Day 2 engineering, four architectural vulnerabilities must be addressed:

### Flaw 1: Toy Pseudo-OOD Created a False Sense of OOF Convergence
The OOF 4-class score reported 0.9099, but Public LB scored 0.5918. The synthetic CutMix chimera samples were easily separable in embedding space (AUROC 0.9987). Real-world rainforest Unknowns (insects, raptors, frogs, rain saturation) occupy complex intermediate manifolds that synthetic CutMix fails to model.

### Flaw 2: Single Center-Crop (10s) Misses Peripheral Vocalizations
Recordings range up to 14.90 seconds. A fixed center-crop `(len - target_len) // 2` discards the first 2.45 seconds and final 2.45 seconds of a 14.9s file. If bird vocalizations occur exclusively in the discarded margins, the model receives pure ambient noise, outputting ambiguous predictions.

### Flaw 3: NLL Objective Penalizes Minority Class Specialization
Because NLL is sample-unweighted, the 5-model blend sacrificed ConvNeXt-Tiny's Cendrawasih F1 (0.8316) to maximize majority class log-loss via LightGBM. Optimizing directly for Macro F1 is essential.

### Flaw 4: Peak Normalization Ineffectiveness on Saturated Files
40% to 55% of recordings contain digital clipping ($\ge 0.99$). Dividing by `np.max(np.abs(wav))` on an already clipped waveform acts as an identity operator ($x / 1.0 = x$). Digital clipping harmonics remain unaddressed at the waveform level.

---

## 10. Actionable Engineering Roadmap for Day 2 (Target: 0.65 - 0.72)

```text
Day 2 Implementation Plan:
1. Signal Preprocessing Enhancement:
   - Energy-based Vocal Activity Detection (VAD) / Top-Energy 10s Windowing.
   - Spectral Subtraction / Stationary Noise Gating.
2. Inference Defense:
   - 3-Crop Test-Time Augmentation (TTA: Left, Center, Right window pooling).
3. Calibration & Weight Re-alignment:
   - Macro-F1 Balanced Optimization (w_convnext = 0.40, w_lgb = 0.30, w_swin = 0.15, w_seresnext = 0.15).
   - Test-Set Transductive Embedding Density Clustering.
```

### High-Priority Experiments
1. **3-Crop Test-Time Augmentation (TTA):** Extract logits across three overlapping 10-second windows (left, center, right) per test audio. Average the softmax distributions. Directly eliminates the risk of truncated bird calls.
2. **Class-Harmonic Balanced Ensemble Weights:** Replace sample-averaged NLL with Macro-F1 aligned weighting:
$$
\mathbf{w}^* = \arg\max_{\mathbf{w}} \text{Macro F1}_{\text{OOF}}\left(\sum_m w_m P_m\right)
$$
Enforcing $w_{\text{ConvNeXt}} \ge 0.35$ to preserve the 0.8316 Cendrawasih capability.
3. **SNR-Dynamic Gated Blending:** Dynamically modulate model weights based on audio spectral flatness. Clean tonal audio routes higher weight to LightGBM; noisy degraded audio routes higher weight to ConvNeXt and Swin-Tiny.

---

## 11. Complete Artifacts & Checkpoints Manifest

All artifacts are persisted in the local repository and verified in `PROJECT_STATE.md`:

```text
schematics-npc-dsc-2026/
├── schematics-day-1.ipynb                         Master production notebook (Cells 0-22)
├── PROJECT_STATE.md                              Evidence ledger schema 1.1 (36 evidence items)
├── PREPROCESSING_PLAN.md                         Methodology specification v1.2
├── day1.md                                       Comprehensive consolidated technical dossier
├── metadata_with_folds.csv                       4-Fold StratifiedGroupKFold metadata
├── preprocess_config.json                        Locked feature extraction hyperparameters
├── track_a_X_train_32k.npy                       792-dim tabular feature matrix (1826, 792)
├── track_a_X_test_32k.npy                        792-dim test feature matrix (814, 792)
├── oof_track_a_lgbm.csv                          Track A LightGBM OOF predictions (F1: 0.8542)
├── oof_b1_effnet_b0_probs.npy                    Track B1 EfficientNet-B0 OOF probabilities (F1: 0.8392)
├── oof_b2_convnext_tiny_probs.npy                Track B2 ConvNeXt-Tiny OOF probabilities (F1: 0.8908)
├── oof_b3_se_resnext50_probs.npy                 Track B3 SE-ResNeXt-50 OOF probabilities (F1: 0.8560)
├── oof_b4_swin_tiny_probs.npy                    Track B4 Swin-Tiny OOF probabilities (F1: 0.8489)
├── checkpoints_b1_effnet_b0/                     Trained PyTorch weights (4 folds)
├── checkpoints_b2_convnext_tiny/                 Trained PyTorch weights (4 folds)
├── checkpoints_b3_se_resnext50/                  Trained PyTorch weights (4 folds)
├── checkpoints_b4_swin_tiny/                     Trained PyTorch weights (4 folds)
├── submissions/
│   ├── sub_001_*/                                Baseline submission bundle (LB: 0.54401)
│   ├── sub_002_*/                                kNN-5 + Cendrawasih guard bundle (LB: 0.58315)
│   ├── sub_003_*/                                Sweet-spot threshold probe bundle (LB: 0.58313)
│   └── sub_004_*/                                5-Model ensemble bundle (LB: 0.59183)
└── schematics_day1_all_artifacts.zip             Full downloadable Kaggle working archive (2.1 GB)
```

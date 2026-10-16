<div align="center">
  <h1>schematics-npc-dsc-2026</h1>
  <p><strong>Tropical Rainforest Bird Chirp Open-Set Classification & Novelty Detection</strong></p>
  
  <p align="center">
    <img src="https://img.shields.io/badge/Competition-Schematics_NPC_DSC_2026-blue?style=flat-square" alt="Competition">
    <img src="https://img.shields.io/badge/Kaggle_Public_LB-0.59183-success?style=flat-square" alt="Leaderboard">
    <img src="https://img.shields.io/badge/Task-Audio_Open--Set_Classification-orange?style=flat-square" alt="Task">
    <img src="https://img.shields.io/badge/Language-Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Language">
    <img src="https://img.shields.io/badge/Status-Active_Phase_1-brightgreen?style=flat-square" alt="Status">
  </p>
  
  <p align="center">
    Autonomous machine learning and deep learning framework for classifying endemic Indonesian bird species and rejecting unseen vocalizations as Unknown under noisy tropical rainforest acoustic dynamics.
  </p>
</div>

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Project Overview

National Data Science Competition (Schematics NPC DSC 2026) hosted by Institut Teknologi Sepuluh Nopember (ITS) Surabaya. The objective is building an acoustic recognition system to assist forest ranger Pak Pisi in monitoring endemic bird species across tropical rainforest sensor deployments.

The system addresses three known classes alongside an unlabelled open-set reject category:
* **Bondol Jawa** (`Lonchura leucogastroides`): 1,124 training samples (61.56%)
* **Kutilang** (`Pycnonotus aurigaster`): 619 training samples (33.90%)
* **Cendrawasih** (`Paradisaeidae`): 83 training samples (4.55%)
* **Unknown**: Target test distribution category with zero training instances.

### Evaluation Criteria

Final preliminary score formula:
$$
\text{Final Score} = 0.35 \times \text{Kaggle Macro F1} + 0.65 \times \text{Technical Report Score}
$$

The technical report carries 65% total weight across five evaluation components: Evaluation and Validation (30%), Modeling (25%), Preprocessing and Feature Engineering (20%), Exploratory Data Analysis (15%), and Structure (10%).

## Empirical Findings & Problem Signatures

Systematic exploration across 1,826 training audio files (48 kHz stereo) revealed critical structural challenges:

1. **Extreme Class Imbalance:** Imbalance ratio $IR = 13.54$ between Bondol Jawa and Cendrawasih.
2. **Recording Hardware Redundancy:** Inter-channel correlation of 0.724 to 0.800 with an average RMS delta of 0.002 supports deterministic mono downmixing `(ch0 + ch1) / 2` without signal loss.
3. **Acoustic Sensor Clipping:** 40% to 55% of recordings exhibit digital clipping ($\ge 0.99$), necessitating polyphase FIR anti-aliasing resampling (`resample_poly`) to prevent Gibbs ringing overshoot.
4. **Spectral Energy Signatures:** Kutilang concentrates 57.2% power in 2 to 4 kHz; Bondol Jawa spans broad bandwidth across 2 to 6 kHz (64.5%); Cendrawasih exhibits a distinct bimodal signature with 44.0% power in 0 to 2 kHz and 14.7% in 6 to 8 kHz.
5. **High-Frequency Auditing:** The ultrasonic band above 16 kHz contains less than 1.22% total energy, while the 8 to 16 kHz band preserves up to 9.08% of Cendrawasih harmonic power, justifying a default target sampling rate of 32 kHz (Nyquist 16 kHz).
6. **Group Dependency Cardinality:** Cendrawasih source groups consist of strictly 4 clusters (sizes 3, 16, 29, 35), capping group-aware cross-validation to 4 folds with mandatory Global Out-of-Fold aggregation.

## Methodology Architecture

The solution implements a 6-phase evidence-aligned machine learning architecture:

### 1. Preprocessing Pipeline
* **Dual-to-Mono Downmixing:** Arithmetic mean `(ch0 + ch1) / 2`.
* **Polyphase Resampling:** 48 kHz to 32 kHz using `scipy.signal.resample_poly` with low-pass FIR filter.
* **Per-Sample Peak Normalization:** Scaling waveform to range `[-1.0, 1.0]` for gain invariance.
* **Reflected Duration Standardization:** Fixed window of 10.0 seconds (320,000 samples at 32 kHz) via `reflect` padding for short audio and center/random cropping for long audio.
* **Log-Mel Spectrogram Extraction:** 128 mel bins, $f_{\min} = 200\text{ Hz}$, $f_{\max} = 16000\text{ Hz}$, `n_fft = 2048`, `hop_length = 512`, `top_db = 80`.
* **Per-Sample Robust Normalization:** Z-score standardization per spectrogram individual.

### 2. Multi-Paradigm Diverse Modeling Portfolio

Five orthogonal model families were trained under a 4-Fold StratifiedGroupKFold validation scheme:

| Track | Family & Backbone | Parameters | Inductive Bias & Role | Global OOF Macro F1 |
|---|---|---|---|---|
| **Track A** | LightGBM Tabular | 792 features | Mel summary stats, temporal deltas, spectral centroid, ZCR | 0.8542 |
| **Track B1** | EfficientNet-B0 | ~5.3M | Depthwise separable conv, compact spatial representation | 0.8392 |
| **Track B2** | ConvNeXt-Tiny | ~28.0M | Modern ConvNet, $7 \times 7$ large kernel, frequency modulation | **0.8908** |
| **Track B3** | SE-ResNeXt-50 32x4d | ~27.5M | Squeeze-and-Excitation channel attention, multi-branch | 0.8560 |
| **Track B4** | Swin-Tiny Transformer | ~28.3M | Shifted-window self-attention, non-local harmonic coupling | 0.8489 |

All neural networks utilize a unified architecture with Linear Probing then Fine-Tuning (LP-FT 2-phase), Class-Balanced Focal Loss ($\gamma = 2.0, \beta = 0.99$), and a 256-dimensional L2-normalized bottleneck projection layer before the classification head.

### 3. Open-Set Rejection & Novelty Detection Engine

Unknown class prediction operates across three distinct detection layers:
* **Layer 1:** Negative Maximum Softmax Probability (MSP).
* **Layer 2:** Free Energy Score $E(x) = -T \cdot \log \sum \exp(z_i / T)$ ($T = 0.6080$).
* **Layer 3:** Fold-Safe Cosine kNN Distance ($k = 5$) on the 256-dimensional bottleneck embedding space (OOD AUROC = 0.9970).
* **Cendrawasih False-Positive Guard:** An asymmetric prototype threshold preventing low-frequency jungle noise and Unknown samples from leaking into the extreme minority class.

## Progression & Submission Ledger

Submissions follow an empirical validation protocol on Kaggle:

| Submission | File | Primary Strategy | Predicted Unknown | Public LB Score | Status |
|---|---|---|---|---|---|
| **sub_001** | `experiment 1.csv` | LightGBM + EffNet-B0 + Global Tau (p95) | 12.04% (98) | 0.54401 | Anchor Baseline |
| **sub_002** | `experiment 2.csv` | Cosine kNN-5 + Cendrawasih Guard + p90 Probe | 36.61% (298) | 0.58315 | +0.03914 jump |
| **sub_003** | `experiment 3.csv` | Sweet-spot p92 Threshold Probe | 28.01% (228) | 0.58313 | Plateau Identified |
| **sub_004** | `experiment 4.csv` | 5-Model Multi-Paradigm Ensemble Blend | 37.71% (307) | 0.59183 | Day 1 Benchmark |
| **sub_005** | `experiment 5.csv` | Preprocessing v2.0 + Dynamic Gating (1-Crop) | 37.84% (308) | 0.55789 | Pure Ide 1 Isolation |
| **sub_006** | `experiment 6.csv` | The Elite Triad + Dynamic Gating + 3-Crop TTA + Prototype Shrinkage | 43.12% (351) | **0.59462** | **New Peak All-Time (+0.03673 vs sub_005)** |

## Repository Structure

```text
schematics-npc-dsc-2026/
├── schematics-day-1.ipynb          Master production notebook containing full pipeline (Cells 1-22)
├── PROJECT_STATE.md               Persistent machine-readable evidence ledger and decisions log
├── PREPROCESSING_PLAN.md          Detailed 6-phase engineering and ablation specifications
├── metadata_with_folds.csv        4-Fold StratifiedGroupKFold metadata partition
├── preprocess_config.json         Locked audio and feature extraction hyperparameters
├── eda_acoustic_profiles.png      3-panel quantitative acoustic analysis visualization
├── eda_mel_spectrograms.png       3-class comparative mel-spectrogram diagnostics
├── schematics_day1_all_artifacts/ Trained model weights, out-of-fold predictions, and metrics
└── submissions/                   Isolated submission bundles with timestamped logs
    ├── sub_001_*/                 Bundle for submission 1 (LB 0.54401)
    ├── sub_002_*/                 Bundle for submission 2 (LB 0.58315)
    ├── sub_003_*/                 Bundle for submission 3 (LB 0.58313)
    └── sub_004_*/                 Bundle for submission 4 (LB 0.59183)
```

## How to Run

1. Clone repository and place dataset into directory `schematics-2026/`:
```bash
git clone https://github.com/ababilkhoerulimam/schematics-npc-dsc-2026.git
cd schematics-npc-dsc-2026
```

2. Install dependencies:
```bash
pip install torch torchvision timm soundfile librosa lightgbm scikit-learn scipy pandas numpy
```

3. Open and run `schematics-day-1.ipynb` sequentially from Cell 1 through Cell 22.

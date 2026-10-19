<div align="center">
  <h1>schematics-npc-dsc-2026</h1>
  <p><strong>Tropical Rainforest Bird Chirp Open-Set Classification & Novelty Detection</strong></p>
  
  <p align="center">
    <img src="https://img.shields.io/badge/Competition-Schematics_NPC_DSC_2026-blue?style=flat-square" alt="Competition">
    <img src="https://img.shields.io/badge/All--Time_Peak_LB-0.66760-success?style=flat-square" alt="All-Time Peak LB">
    <img src="https://img.shields.io/badge/Day_4_Champion_LB-0.65993-blue?style=flat-square" alt="Day 4 Champion LB">
    <img src="https://img.shields.io/badge/Task-Audio_Open--Set_Classification-orange?style=flat-square" alt="Task">
    <img src="https://img.shields.io/badge/Language-Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Language">
    <img src="https://img.shields.io/badge/Status-Active_Phase_1-brightgreen?style=flat-square" alt="Status">
  </p>
  
  <p align="center">
    Autonomous bioacoustic machine learning and deep learning framework for classifying endemic Indonesian bird species and rejecting unseen vocalizations as Unknown under noisy tropical rainforest acoustic dynamics.
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
7. **Recorder Confounder Size:** Kruskal-Wallis testing proves that `source_group` accounts for $\eta^2 = 0.2262$ (22.62%) of signal variance on Bondol Jawa ($H = 258.66, p = 5.77 \times 10^{-53}$), enforcing strict GroupKFold partitioning to prevent source leakage.

## Methodology Architecture

The solution implements an evidence-aligned machine learning architecture across multiple model families:

### 1. Preprocessing Pipeline
* **Dual-to-Mono Downmixing:** Arithmetic mean `(ch0 + ch1) / 2`.
* **Polyphase Resampling:** 48 kHz to 32 kHz using `scipy.signal.resample_poly` with low-pass FIR filter.
* **Butterworth High-Pass Filtering:** 4th-order filter at 100 Hz cutoff to suppress sensor handling rumble while preserving deep Cendrawasih vocalizations.
* **Per-Window Linear Peak Normalization:** Linear scaling to target peak 0.95 without non-linear tanh distortion to protect bioacoustic PCEN features.
* **Dynamic Equidistant 3-Window Slicing:** Extracts three 5.0-second windows (160,000 samples @ 32 kHz for Google Perch) and three 3.0-second windows (144,000 samples @ 48 kHz for BirdNET).

### 2. Multi-Paradigm Diverse Modeling Portfolio

Multiple orthogonal model families were benchmarked under a 4-Fold StratifiedGroupKFold validation scheme:

| Track | Family & Backbone | Input Representation | Inductive Bias & Role | Global OOF Macro F1 |
|---|---|---|---|---|
| **Pillar 1** | LightGBM + XGBoost | 818 DSP Features | Signal physics (ZCR, spectral centroid, roll-off, MFCCs) | 0.8463 |
| **Pillar 2** | ConvNeXt-Tiny v2.0 | 128-bin Log-Mel Spectrogram | 2D Spatial texture, $7 \times 7$ kernels, SpecAugment, Focal Loss | 0.9323 |
| **Pillar 3** | Google Perch v2 | 1536-d Bioacoustic Embedding | 14,795 worldwide taxa bioacoustic pre-training | 0.9283 |
| **Day 4 SOTA** | Transductive Hybrid Perch | 1543-d Hybrid Input (1536-d + 7-d Summary) | In-fold self-training with 99 verified test inliers + forensic reassignment | **0.9515** |

### 3. Open-Set Rejection & Forensic Shielding Engine

Unknown class prediction and novelty detection operate across three integrated layers:

1. **Forensic Taxonomic Reassignment:** Soft reassignment of Perch Cendrawasih calls to the second-highest class when BirdNET MaxCendra logit falls below the in-fold knee-point threshold ($\tau \approx -7.70$). This removes false alarms on noisy audio and jumps OOF Macro F1 from 0.9119 to 0.9383 on supervised models and to 0.9515 on self-trained models.
2. **Forensic Alien Bulbul Isolation:** Isolates Yellow-vented Bulbul (`Pycnonotus goiavier` / Trucuk) from Kutilang (`Pycnonotus aurigaster`) when Perch confidence is marginal ($P < 0.60$) and BirdNET Kutilang detector is inactive ($\tau \approx -8.11$).
3. **Manifold Out-of-Distribution Calibration:** In-fold mean-centered kNN-5 cosine distance calibrated to the sweet-spot Unknown rate ($\approx 43.5\%$), protecting core endemic bird recall while rejecting novel background soundscapes.

## Progression & Submission Ledger

Submissions follow an empirical validation protocol on the Kaggle Leaderboard:

| Submission | File | Primary Strategy | Predicted Unknown | Public LB Score | Status |
|---|---|---|---|---|---|
| **sub_001** | `experiment 1.csv` | LightGBM + EffNet-B0 + Global Tau (p95) | 12.04% (98) | 0.54401 | Anchor Baseline |
| **sub_002** | `experiment 2.csv` | Cosine kNN-5 + Cendrawasih Guard + p90 Probe | 36.61% (298) | 0.58315 | +0.03914 jump |
| **sub_003** | `experiment 3.csv` | Sweet-spot p92 Threshold Probe | 28.01% (228) | 0.58313 | Plateau Identified |
| **sub_004** | `experiment 4.csv` | 5-Model Multi-Paradigm Ensemble Blend | 37.71% (307) | 0.59183 | Day 1 Benchmark |
| **sub_005** | `experiment 5.csv` | Preprocessing v2.0 + Dynamic Gating (1-Crop) | 37.84% (308) | 0.55789 | Pure Ide 1 Isolation |
| **sub_006** | `experiment 6.csv` | The Elite Triad + Dynamic Gating + 3-Crop TTA | 43.12% (351) | 0.59462 | Day 2 Peak Benchmark |
| **sub_007** | `experiment 7.csv` | Dual-Manifold kNN + Bioacoustic Gate (70 Cendra) | 47.54% (387) | 0.58805 | Over-rejection boundary |
| **sub_008** | `experiment 8.csv` | Golden Sweet-Spot Interpolation (82 Cendra) | 43.86% (357) | 0.58356 | Global Tau p92 probe |
| **sub_009** | `experiment 9.csv` | Google Perch v2 Bioacoustic Standalone (1536-d) | 32.92% (268) | 0.63967 | Day 3 Perch Breakthrough |
| **sub_010** | `experiment 10.csv` | 3-Pillar Master Ensemble + Pure ECDF OOD | 55.28% (450) | 0.63256 | Conservative Multi-Manifold |
| **sub_011** | `experiment 11.csv` | Google Perch v2 Pure + Calibrated ~40% Unknown | 40.05% (326) | 0.65233 | 0.65+ Breakthrough |
| **sub_012** | `experiment 12.csv` | Perch v2 + BirdNET Forensic Reassignment + Shield | 43.86% (357) | 0.65447 | Forensic Alignment Peak |
| **sub_013** | `experiment 13.csv` | Perch v2 + Adaptive Self-Training (126 Pseudo) | 43.49% (354) | **0.66760** | **All-Time Peak Benchmark** |
| **sub_014** | `experiment 14.csv` | Pure Supervised Hybrid (1543-d) + Forensic Shield | 43.49% (354) | 0.64396 | Day 4 Safe Anchor Baseline |
| **sub_015** | `experiment 15.csv` | In-Fold Self-Trained Hybrid (1543-d) @ 43.5% | 43.49% (354) | **0.65993** | **Day 4 SOTA Champion (+0.01597 vs sub_014)** |
| **sub_016** | `experiment 16.csv` | In-Fold Self-Trained Hybrid (1543-d) @ 41.5% | 41.52% (338) | 0.65052 | Conservative Probe (-0.00941 drop) |
| **sub_017** | `experiment 17.csv` | Class-Conditional Manifold Spheres (p95 per class) | 38.45% (313) | 0.57045 | Intrinsic OOD (False Positive Bleed) |
| **sub_018** | `experiment 18.csv` | Pure Global OOF Inlier Retention (p95 tau=0.6106) | 35.75% (291) | 0.63049 | Intrinsic OOD (Under-rejection) |
| **sub_019** | `experiment 19.csv` | In-Fold Self-Trained (203 Inliers) + BirdNET Shield @ 43.5% | 43.49% (354) | 0.67359 | Day 5 SOTA (+0.02663 vs sub_021) |
| **sub_020** | `experiment 20.csv` | In-Fold Self-Trained (203 Inliers) Probe Low @ 42.0% | 42.01% (342) | 0.65818 | Probe Low (-0.01541 drop) |
| **sub_021** | `experiment 21.csv` | Pure Supervised Hybrid Anchor (81 Cendra) @ 43.5% | 43.49% (354) | 0.64696 | Day 5 Safe Anchor Baseline |
| **sub_022** | `experiment 22.csv` | In-Fold Self-Trained (203 Inliers) Probe High @ 45.0% | 44.96% (366) | **0.67701** | **ALL-TIME PEAK RECORD (+0.00941 vs sub_013)** |

## Repository Structure

```text
schematics-npc-dsc-2026/
├── notebooks/
│   ├── schematics-day-1.ipynb          Master production notebook Day 1 (Cells 1-23)
│   ├── schematics-day-2.ipynb          Master production notebook Day 2 (Cells 1-28, Peak LB 0.59462)
│   ├── schematics-day-3.ipynb          Master production notebook Day 3 (Cells 1-32, Peak LB 0.66760)
│   ├── schematics-day-4.ipynb          Master production notebook Day 4 (Cells 1-19, Champion LB 0.65993)
│   ├── notebook_jere.ipynb             EDA & Kruskal-Wallis confounder analysis notebook
│   ├── notebook_jere.md                Markdown transcript of Jere analysis
│   └── ANALYSIS_STATE.md               State ledger for exploratory confounder analysis
├── schematics_day1_all_artifacts/      Day 1 artifacts, planning day 1, plots, metrics, submissions 1-4
│   ├── planning day 1.md               Day 1 planning and ablation blueprint
│   ├── day1.md                         Day 1 comprehensive technical dossier
│   ├── schematics_day1_notebook.md     Markdown export of Day 1 notebook
│   ├── eda_acoustic_profiles.png       3-panel quantitative acoustic analysis visualization
│   ├── eda_mel_spectrograms.png        3-class comparative mel-spectrogram diagnostics
│   └── submissions/                    Submissions 1 to 4 bundles (sub_001 to sub_004)
├── schematics_day2_all_artifacts/      Day 2 artifacts, planning day 2, plots, metrics, submissions 5-8
│   ├── planning day 2.md               Day 2 28-cell technical master blueprint
│   ├── eda_day2_preprocessing_v2_verification.png Preprocessing v2.0 verification spectrogram
│   ├── model_grand_prix_summary.json   7-model decision matrix and F1 benchmarks
│   ├── dynamic_gating_summary.json     SNR-Dynamic Gating calibration parameters
│   └── submissions/                    Submissions 5 to 8 bundles (sub_005 to sub_008)
├── schematics_day3_all_artifacts/      Day 3 artifacts, planning day 3, plots, metrics, submissions 9-13
│   ├── planning_day_3.md               Day 3 technical master blueprint
│   ├── ood_dual_manifold_diagnostics.png Dual-manifold distance space and bimodal OOD score plots
│   ├── ood_thresholds.json             Dual-manifold statistical calibration parameters
│   ├── dynamic_gating_summary.json     3-Pillar gating parameters and SNR noise tersil proof
│   └── submissions/                    Submissions 9 to 13 bundles (sub_009 to sub_013)
├── schematics_day4_all_artifacts/      Day 4 artifacts, modular master blueprint, transductive self-training
│   ├── planning_day_4.md               Day 4 36-cell modular execution blueprint v3.0
│   ├── planning_day_4_experiment.md    Day 4 lean 17-cell rapid execution blueprint v2.0
│   ├── metadata_with_folds.csv         Verified 4-Fold StratifiedGroupKFold metadata
│   ├── nested_forensic_thresholds.json In-fold Youden knee-point calibration parameters
│   ├── curated_inlier_meta.json        Consensus 99 inlier curation metadata (99.41% purity)
│   ├── dual_forensics_meta.json        Supervised vs Self-Trained cross-branch agreement (91.52%)
│   ├── quantile_locking_meta.json      Sweet-spot manifold quantile locking parameters
│   ├── download_day4_submissions.html  Self-contained browser-native submission downloader
│   └── submissions/                    Submissions 14 to 18 bundles (sub_014 to sub_018)
├── papers/                             Curated open-set audio literature and paper directory
│   ├── README.md                       Literature review and mathematical taxonomy
│   ├── Energy_based_OOD_detection_2010.03759.pdf NeurIPS 2020 foundation paper on Free Energy
│   ├── Multi_label_open_set_audio_classification_2310.13759.pdf DCASE 2023 open-set audio taxonomy
│   └── ROLE_few_shot_open_set_audio_classification_2607.26607.pdf SCUT 2026 transductive prototype refinement
├── PROJECT_STATE.md                    Persistent machine-readable evidence ledger and decisions log (E-001 to E-094)
├── LICENSE                             Repository license
└── README.md                           Publication-grade documentation
```

## How to Run

1. Clone repository and place dataset into directory `schematics-2026/`:
```bash
git clone https://github.com/ababilkhoerulimam/schematics-npc-dsc-2026.git
cd schematics-npc-dsc-2026
```

2. Install dependencies:
```bash
pip install torch torchvision timm soundfile librosa lightgbm xgboost catboost scikit-learn scipy pandas numpy tensorflow tensorflow-hub birdnet
```

3. Open and run master notebooks inside directory `notebooks/` sequentially:
```bash
# Day 1 Baseline and Foundation Exploration
jupyter notebook notebooks/schematics-day-1.ipynb

# Day 2 Master Pipeline (Peak LB 0.59462)
jupyter notebook notebooks/schematics-day-2.ipynb

# Day 3 Bioacoustic Foundation & Forensic Filtering (All-Time Peak LB 0.66760)
jupyter notebook notebooks/schematics-day-3.ipynb

# Day 4 Transductive In-Fold Self-Training & Intrinsic OOD (Day 4 Champion LB 0.65993)
jupyter notebook notebooks/schematics-day-4.ipynb
```

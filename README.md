<div align="center">
  <img src="logo.png" alt="Logo" width="128" height="128">
  
  <h1>schematics-npc-dsc-2026</h1>
  <p><strong>Tropical Rainforest Bird Chirp Open-Set Classification and Novelty Detection</strong></p>
  
  <p align="center">
    <img src="https://img.shields.io/badge/Competition-Schematics_NPC_DSC_2026-blue?style=flat-square" alt="Competition">
    <img src="https://img.shields.io/badge/Historic_Peak_LB-0.71221-success?style=flat-square" alt="Historic Peak LB">
    <img src="https://img.shields.io/badge/Master_Pipeline_OOF-0.9468-brightgreen?style=flat-square" alt="Master Pipeline OOF">
    <img src="https://img.shields.io/badge/Task-Audio_Open--Set_Classification-orange?style=flat-square" alt="Task">
    <img src="https://img.shields.io/badge/Language-Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Language">
    <img src="https://img.shields.io/badge/Status-Completed_Reproducible-success?style=flat-square" alt="Status">
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

National Data Science Competition (Schematics NPC DSC 2026) organized by Institut Teknologi Sepuluh Nopember (ITS) Surabaya. The objective is developing an acoustic recognition framework to assist forest ranger Pak Pisi in monitoring endemic bird species across rainforest sensor deployments.

The framework classifies three endemic species alongside an unlabelled open-set category:
* Bondol Jawa (`Lonchura leucogastroides`): 1,124 training samples (61.56%)
* Kutilang (`Pycnonotus aurigaster`): 619 training samples (33.90%)
* Cendrawasih (`Paradisaeidae`): 83 training samples (4.55%)
* Unknown: Target test distribution category containing unlabelled avian and non-avian sounds with zero training instances.

### Evaluation Criteria

Final competition score formula:

$$
\text{Final Score} = 0.35 \times \text{Kaggle Macro F1} + 0.65 \times \text{Technical Report Score}
$$

The technical report carries 65% total weight across five components: Evaluation and Validation (30%), Modeling (25%), Preprocessing and Feature Engineering (20%), Exploratory Data Analysis (15%), and Structure (10%).

## Key Empirical Findings

Exploration and causal validation across 1,826 training audio recordings (48 kHz stereo) established the following empirical evidence:

1. Extreme Class Imbalance: An imbalance ratio of 13.54 exists between Bondol Jawa and Cendrawasih, requiring Class-Balanced Focal Loss with smoothing.
2. Hardware Recorder Confounder: Kruskal-Wallis testing proves recording hardware accounts for 23.66% of acoustic signal variance on Bondol Jawa (H = 270.32, p = 1.86e-55), demonstrating that random K-Fold creates severe leakage gaps (+0.0642 artificial score inflation).
3. Nyquist Frequency Cutoff: Cendrawasih vocalizations exhibit an essential bimodal harmonic signature with 16.5% power in 6 to 8 kHz and 9.08% in 8 to 16 kHz. Resampling to 16 kHz (Nyquist 8 kHz) causes 45.8% False Negatives on Cendrawasih (F1 drops to 0.3704), eliminating 16 kHz Audio Transformers like Bird-MAE and BEATs.
4. Active-RMS Normalization: Digital clipping affects 40% to 55% of recordings, rendering peak normalization destructive. Active-frame RMS normalization at -18 dBFS preserves bioacoustic PCEN features.
5. Invariant Latent Space: Google Perch v2 1536-d embeddings project vocalizations onto a shared biological manifold, suppressing microphone fingerprints to near zero while maintaining wide inter-class separation.
6. Orthogonal Forensic Shielding: BirdNET v2.4 (48 kHz native) acts as an independent auditor to veto false positive Cendrawasih calls and isolate Yellow-vented Bulbul (`Pycnonotus goiavier` / Trucuk) vocalizations.
7. Transductive Saturation Point: In-fold self-training reaches optimal Pareto convergence at Round 2 (280 inliers, lambda = 0.85). Round 3 degrades F1 by -0.0074 due to high-frequency insect stridulation (Cicadidae) intrusion.
8. Orthogonal Error Synergy: An inlier ensemble combining 65% Perch v2 and 35% ConvNeXt-Tiny resolves 21 out of 28 Perch errors, reaching a peak OOF Macro F1 of 0.9468.

## Master Pipeline Architecture (10 Phases)

The end-to-end framework in `notebooks/DatSci_disuruh ryu daftar_Notebook.ipynb` implements 65 structured cells across 10 deterministic phases:

1. Environment and VRAM Optimization: Dynamic GPU growth for TensorFlow Hub alongside CuDNN deterministic execution under Seed 42.
2. Bioacoustic Exploratory Data Analysis: Statistical distributions, Kolmogorov-Smirnov duration shift tests, and Kruskal-Wallis confounder forensics.
3. Preprocessing and Feature Engineering: 32 kHz polyphase resampling, 300 Hz Butterworth HPF, active-RMS normalization, and canonical 818-d DSP feature extraction.
4. Conventional Modeling Grand Prix: Tabular benchmarks (CatBoost F1 = 0.8082), 2D Vision benchmarks (ConvNeXt-Tiny F1 = 0.9043), AST 16 kHz failure proof, and dynamic SNR gating (F1 = 0.9009 ceiling).
5. Bioacoustic Foundation Models: Google Perch v2 1536-d latent extraction, Multi-Taxa Paradisaeidae Max-Pooling across 8 taxa, and 1543-d Hybrid CosineHead calibration.
6. Forensic Validation and Dual Shielding: In-fold BirdNET v2.4 threshold calibration (tau = -7.95) to veto Cendrawasih false alarms and isolate Trucuk calls.
7. Transductive Self-Training: 4-Gate sterile inlier curation (99.40% purity), Round 1 adaptation, Round 2 Pareto peak (F1 = 0.9381), and Round 3 saturation proof.
8. Decision Gate Optimization: Architectural ablation (Gate A), mixup augmentation (Gate B adopted), pseudo-label weight sweep (Gate C), and orthogonal inlier ensembling (Gate D adopted).
9. Grand Synthesis and Dual-Organic Delivery: End-to-end retrained assembly producing Slot A (Geometric Manifold OOD), Slot B (Taxonomic Consensus OOD), and the Hybrid Champion (`sub_ensemble09468_geo371_rev.csv`).
10. Delivery and Reproducibility Audit: 11-phase master ablation ledger, SHA-256 cryptographic manifest for 11 key artifacts, and automated ZIP archival.

## Submission Progression Ledger

| Submission | File | Primary Strategy | Unknown Rate | Public LB Score | Role and Status |
|:---|:---|:---|:---:|:---:|:---|
| sub_001 | `experiment 1.csv` | LightGBM + EffNet-B0 + Global Tau (p95) | 12.04% (98) | 0.54401 | Anchor Baseline |
| sub_006 | `experiment 6.csv` | The Elite Triad + Dynamic Gating + 3-Crop TTA | 43.12% (351) | 0.59462 | Day 2 Peak Benchmark |
| sub_013 | `experiment 13.csv` | Perch v2 + Adaptive Self-Training (126 Pseudo) | 43.49% (354) | 0.66760 | Day 3 Breakthrough |
| sub_015 | `experiment 15.csv` | In-Fold Self-Trained Hybrid (1543-d) @ 43.5% | 43.49% (354) | 0.65993 | Day 4 Champion |
| sub_019 | `experiment 19.csv` | In-Fold Self-Trained (203 Inliers) + BirdNET Shield | 43.49% (354) | 0.67359 | Day 5 Round 1 Base |
| sub_022 | `experiment 22.csv` | In-Fold Self-Trained (203 Inliers) High Probe @ 45.0% | 44.96% (366) | 0.67701 | Round 1 High Probe |
| sub_023 | `experiment 23.csv` | Round-2 Self-Trained (274 Inliers) Multi-Taxa Pooled | 44.96% (366) | 0.71221 | Historic All-Time Peak SOTA |
| sub_024 | `experiment 24.csv` | Round-3 Self-Trained (292 Inliers) Multi-Taxa Pooled | 44.96% (366) | 0.69098 | Round 3 Saturation Proof |
| sub_r2_shield | `sub_selftrain_r2_shielded_rev.csv` | Master Pipeline Round 2 (280 Inliers) + Shield | 45.58% (371) | 0.69264 | Master Pipeline R2 Anchor |
| sub_opt_geo | `sub_optimized_geometric_rev.csv` | Inlier Ensemble (w=0.65) + Manifold OOD (p95) | 22.60% (184) | 0.61681 | Champion Slot A (Conservative) |
| sub_opt_tax | `sub_optimized_taxonomic_rev.csv` | Inlier Ensemble (w=0.65) + Taxonomic OOD (p10xp5) | 43.24% (352) | 0.61766 | Champion Slot B (Consensus) |
| sub_hybrid_champion | `sub_forensic_shielded_rev.csv` | Inlier Ensemble (w=0.65) + Sub 3 Geometric Cutoff (tau=0.5567) | 45.58% (371) | - | Hybrid Champion (Ens 0.9468 + 45.6% Geo) |

## Final Submission Selection Strategy

Candidate options for the two final submission slots on Kaggle:

1. Option 1 (Historical SOTA Peak): `sub_023` (Public LB 0.71221). The historical peak submission utilizing Round 2 self-training with 274 curated inliers and Multi-Taxa Paradisaeidae pooling.
2. Option 2 (Reproducible Master Anchor): `sub_selftrain_r2_shielded_rev.csv` (Public LB 0.69264). Generated from the verified master pipeline with 280 inliers, 45.58% organic unknown proportion, and zero group leakage.
3. Option 3 (Optimal Hybrid Ensemble): `sub_forensic_shielded_rev.csv`. Combines the superior species discrimination of the Champion Inlier Ensemble (OOF Macro F1 = 0.9468) with the empirically proven 45.58% Unknown geometric cutoff (tau = 0.5567), correcting 15 inlier samples over Sub 3.

## Repository Structure

```text
schematics-npc-dsc-2026/
├── notebooks/
│   └── DatSci_disuruh ryu daftar_Notebook.ipynb  Master end-to-end pipeline (65 cells, fully annotated)
├── reports/
│   ├── DatSci_disuruh ryu daftar_Laporan.pdf     Final technical report (10 pages)
│   └── DatSci_disuruh ryu daftar_Laporan.docx    Editable source technical report
├── submissions/
│   ├── sub_selftrain_r2_shielded_rev.csv         Master pipeline R2 submission (LB 0.69264)
│   ├── sub_forensic_shielded_rev.csv             Forensic shielded submission
│   ├── sub_optimized_geometric_rev.csv           Champion Slot A submission CSV
│   ├── sub_optimized_taxonomic_rev.csv           Champion Slot B submission CSV
│   ├── sub_perch_hybrid_baseline_rev.csv         Baseline hybrid submission CSV
│   └── submission_comparison_report_rev.txt      Submission distribution report
├── artifacts/
│   ├── figures/                                  26 publication-grade figures (DPI 150)
│   └── metadata/                                 Ablation ledgers, gates, sha256 manifests
├── models/                                       Local model weights (.pt) and probability arrays (.npy)
├── docs/
│   ├── master.md                                 Comprehensive research dossier and pipeline specification
│   ├── final_planning.md                         Master research blueprint specification
│   └── PROJECT_STATE.md                          Empirical evidence ledger (E-001 to E-141)
├── papers/                                       Curated open-set bioacoustic literature
│   └── README.md
├── logo.png                                      Project logo
├── LICENSE                                       MIT License
└── README.md                                     Repository documentation
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

3. Execute the master pipeline notebook:
```bash
jupyter notebook notebooks/DatSci_disuruh ryu daftar_Notebook.ipynb
```

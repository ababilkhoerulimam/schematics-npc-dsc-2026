# Planning Day 2: Audio Preprocessing v2.0, Dual-Tree Tabular Engine, and Multi-View Open-Set Architecture

**Project:** Schematics NPC DSC 2026 - Bird Chirp Openset Classification  
**Target:** Advance Public Leaderboard from 0.59183 toward 0.6800 - 0.7200  
**Version:** 2.0 (Day 2 Blueprint)  
**Date:** 2026-10-16  
**Status:** Approved for Execution  

---

## 1. Executive Summary & Day 2 Objectives

Day 1 concluded with a top Public Leaderboard score of **0.59183** (Rank 17, `sub_004`), proving that multimodal ensembling (Tabular LightGBM + ConvNeXt-Tiny + Swin-Tiny) and open-set kNN-5 rejection with a Cendrawasih guard outperform standard baselines.

However, diagnostic post-mortem revealed that post-processing threshold adjustments have reached an empirical ceiling (plateau between 0.58313 and 0.59183). Advancing into the 0.65+ to 0.72 bracket requires resolving three core physical signal bottlenecks in preprocessing, upgrading the tabular representation, and reforming ensemble optimization from sample-unweighted NLL to Macro-F1 alignment.

### Primary Day 2 Targets
1. **Physical Preprocessing Overhaul (v2.0):** Implement Energy-Based Vocal Activity Windowing (VAD), Stationary Spectral Subtraction, and Active-Frame RMS normalization to eliminate silent-segment cropping and digital clipping saturation.
2. **Dual-Tree Tabular Engine:** Expand tabular features from 792 to 818 dimensions with high-harmonic bimodal ratios (up to 16 kHz) and rhythmic autocorrelation. Train LightGBM alongside XGBoost and evaluate a 50-50 leaf-wise plus depth-wise tree blend.
3. **Deep Learning Retraining on Cleaned Manifolds:** Retrain ConvNeXt-Tiny (B2) and Swin-Tiny (B4) using Energy-Biased Jittered Cropping. Retain Day 1 SE-ResNeXt-50 (B3) for cross-preprocessing multi-view robustness.
4. **SNR-Dynamic Gated Blending:** Replace static probability averaging with an audio-degradation gating function that shifts weight from tabular features (high SNR) to spatial deep learning (low SNR).
5. **3-Crop Test-Time Augmentation (TTA):** Replace deterministic center-cropping at test time with left-center-right 10-second multi-window pooling.

---

## 2. Post-Mortem Day 1: Root Causes of the 0.59183 Ceiling

Detailed audit of Day 1 artifacts (`schematics_day1_all_artifacts/`) exposed four critical weaknesses:

1. **Blind Center-Cropping Missed Peripheral Calls (Flaw 2):** Recordings reach 14.90 seconds. A deterministic center-crop `(len - 10s) // 2` discards the first 2.45s and final 2.45s. For recordings where birds vocalized only in the opening or closing seconds, models received pure ambient jungle noise, forcing ambiguous predictions.
2. **Peak Normalization Failure Under Digital Clipping (Flaw 4):** 55% of test recordings exhibit digital clipping ($\ge 0.99$, with peaks up to 1.004). Dividing by `np.max(np.abs(wav))` on saturated signals acts as an identity function ($x / 1.0 = x$). Low-amplitude bird calls within clipped recordings remained compressed near the noise floor.
3. **Unweighted NLL Suppressed Minority Class Weights (Flaw 3):** Optimizing standard multi-class cross-entropy on Out-of-Fold predictions biased weights toward majority classes (Bondol and Kutilang comprise 95.45% of train data). ConvNeXt-Tiny, which achieved standalone Cendrawasih F1 of 0.8316, was reduced to 0.207 weight, dropping ensemble Cendrawasih F1 to 0.6857.
4. **Low-Frequency Noise Contamination in Cendrawasih:** Ambient rainforest rumble ($0\text{--}300\text{ Hz}$) in the test set mimicked Cendrawasih low-frequency energy (44.0% in $0\text{--}2\text{ kHz}$), triggering 175 raw false-positive predictions in `sub_004` that required aggressive post-hoc pruning.

---

## 3. Fase 1: Signal Preprocessing v2.0 (Physical Corrections)

### 1.1 Acoustic Energy Saliency Windowing (VAD)
Instead of static center-cropping, calculate a sliding 10.0-second window (320,000 samples at 32 kHz) with maximum RMS energy inside the primary avian vocalization band ($2\text{--}8\text{ kHz}$):

* **For Tabular Feature Extraction & Validation Evaluation:** Use 100% deterministic Top-Energy VAD Window. The 10-second segment with the highest vocal-band energy is selected.
* **For Deep Learning Training (Anti-Overfitting Jitter Guard):** Avoid deterministic static cropping during training, which eliminates stochastic regularization. Identify the peak vocal center $t_{\text{peak}}$, then permit the training window to jitter randomly by $\pm 1.5\text{ seconds}$ around $t_{\text{peak}}$ across epochs. This guarantees the bird chirp is always present in the crop while preventing the network from memorizing fixed temporal coordinates.

### 1.2 Stationary Spectral Subtraction with Half-Wave Rectification
Tropical rainforest field sensors record continuous background hiss from rain, running water, and nocturnal insects:

* Compute the 10th percentile noise floor across time frames for each mel bin on the uncorrupted spectrogram:
$$
P_{10}(f) = \text{Percentile}_{10}\left(S_{\text{dB}}(f, t), \text{axis}=t\right)
$$
* Subtract the stationary background and apply half-wave rectification in the decibel domain:
$$
S_{\text{clean}}(f, t) = \max\left(S_{\text{dB}}(f, t) - P_{10}(f), 0.0\right)
$$
* **Pipeline Order Invariant:** Spectral subtraction is executed directly after Log-Mel extraction and before SpecAugment or Gaussian noise injection. This prevents subtraction from neutralizing injected training noise, and half-wave rectification prevents negative outliers from corrupting subsequent per-sample Z-score normalization.

### 1.3 Active-Frame RMS Normalization & Hardware Clipping Recovery
* **Active-Frame Normalization:** Compute signal RMS exclusively over active frames where amplitude exceeds $-40\text{ dBFS}$ ($|x| > 0.01$). Scale the waveform such that active RMS equals $0.12$, followed by a soft $\tanh$ saturation limiter to prevent digital clipping spikes from dominating the dynamic range:
$$
x_{\text{scaled}} = \tanh\left(\frac{x}{\text{RMS}_{\text{active}} + 1e-6} \times 0.12\right)
$$
* **Hardware Distortion Features:** Extract explicit distortion metrics (`clipping_ratio` = percentage of samples with $|x| \ge 0.99$, and `crest_factor` = peak amplitude divided by active RMS) into the tabular matrix so decision trees recognize distorted recordings.

### 1.4 High-Pass 300 Hz Cutoff
Apply a 4th-order Butterworth high-pass filter at 300 Hz (or set $f_{\min} = 300\text{ Hz}$ on mel filters) to suppress microphone wind buffeting without truncating bird vocalizations.

---

## 4. Fase 2: Upgraded Tabular Feature Matrix (~818 Dims) & Dual-Tree Ensemble

### Feature Matrix Enhancements (from 792 to ~818 features)
1. **Three High-Harmonic Bimodal Ratios (Capturing Cendrawasih up to 16 kHz):**
   * `ratio_low_to_mid`: $E_{0.3\text{--}2\text{kHz}} / (E_{2\text{--}6\text{kHz}} + \epsilon)$ (high for Cendrawasih, low for Kutilang).
   * `ratio_high_harmonic`: $E_{8\text{--}16\text{kHz}} / (E_{0.3\text{--}8\text{kHz}} + \epsilon)$ (differentiates Cendrawasih 9.08% harmonic energy from low-frequency wind rumble).
   * `bimodal_product`: $\sqrt{E_{0.3\text{--}2\text{kHz}} \times E_{8\text{--}16\text{kHz}}} / (E_{\text{total}} + \epsilon)$ (peaks exclusively when low rumble and ultrasonic overtones co-occur).
2. **Harmonic-Temporal Rhythm Index:** Autocorrelation of dominant mel bins across lags of 0.1s to 0.8s, separating rhythmic biological chirps from continuous abiotic noise (rain, wind).
3. **Distortion Descriptors:** `clipping_ratio` and `crest_factor`.

### Dual-Tree Head-to-Head & Blending
* **Model 1: LightGBM:** Leaf-wise tree growth, `class_weight='balanced'`, 1000 estimators, learning rate 0.03.
* **Model 2: XGBoost:** Depth-wise tree growth (`max_depth=6`), GPU histogram tree method, balanced sample weights.
* **Ensemble Strategy:** Rather than discarding the runner-up, combine 50% LightGBM + 50% XGBoost probabilities. Blending leaf-wise and depth-wise decision boundaries reduces variance on extreme minority classes.

---

## 5. Fase 3: Deep Learning Retraining with Preprocessing v2.0

### Model Training Focus
1. **Track B2 (ConvNeXt-Tiny, ~28.0M params):** Retrained with Energy-Biased Jittered VAD input and spectral subtraction. Remains the primary backbone due to its 0.8316 Cendrawasih capability.
2. **Track B4 (Swin-Tiny Transformer, ~28.3M params):** Retrained on cleaned spectrograms. Shifted-window self-attention operates with higher fidelity once stationary noise streaks are eliminated.
3. **Cross-Preprocessing Multi-View Retention:** Retain checkpoints and OOF probabilities of Day 1 **SE-ResNeXt-50 (Track B3)** and **EfficientNet-B0 (Track B1)**. Ensembling models trained on raw Log-Mel (v1.0) with models trained on spectrally subtracted Log-Mel (v2.0) introduces structural data-view diversity that protects against spectral subtraction artifacts.

---

## 6. Fase 4: Closed-Set Optimization & SNR-Dynamic Gated Blending

### Macro-Weighted Loss Alignment
Replace unweighted NLL optimization with Class-Harmonic NLL optimization:
$$
\mathcal{L}_{\text{macro}}(\mathbf{w}) = -\frac{1}{C} \sum_{c=1}^C \frac{1}{N_c} \sum_{i \in \text{Class } c} \log\left(\sum_m w_m P_m(y_i = c | x_i)\right)
$$
This ensures Cendrawasih ($N=83$) contributes equal weight to the optimization objective as Bondol Jawa ($N=1124$), maintaining ConvNeXt-Tiny weight at $\ge 0.35$.

### SNR-Dynamic Gated Blending Curve
Calculate an acoustic signal quality index $\text{SNR}_{\text{proxy}}(x_i) \in [0, 1]$ based on Spectral Flatness ($\text{SFM}$) and active RMS:
$$
\text{SNR}_{\text{proxy}}(x_i) = \text{RankNorm}\left(\frac{\text{RMS}_{\text{active}}(x_i)}{\text{RMS}_{\text{floor}}(x_i) + \epsilon} - 2.0 \times \text{SFM}(x_i)\right)
$$

Interpolate weights dynamically per sample:
$$
w_{\text{Tabular}}(x_i) = w_{\min} + (w_{\max} - w_{\min}) \cdot \left(\text{SNR}_{\text{proxy}}(x_i)\right)^\gamma
$$
$$
w_{\text{DL}}(x_i) = 1.0 - w_{\text{Tabular}}(x_i)
$$

* **High-SNR Audio ($\text{SNR}_{\text{proxy}} \to 1$):** Tabular weight rises to 0.45 - 0.50, leveraging LightGBM/XGBoost precision on clean whistles.
* **Low-SNR Audio ($\text{SNR}_{\text{proxy}} \to 0$):** Deep learning weight rises to 0.85 - 0.90, utilizing ConvNeXt-Tiny spatial noise invariance.

---

## 7. Fase 5: 3-Crop Test-Time Augmentation (TTA) & Cascaded OOD Rejection

### 3-Crop Multi-Window TTA
For test recordings exceeding 10.0 seconds:
1. Extract Window Left ($0.0\text{s} \to 10.0\text{s}$).
2. Extract Window Center ($t_{\text{mid}} - 5.0\text{s} \to t_{\text{mid}} + 5.0\text{s}$).
3. Extract Window Right ($L - 10.0\text{s} \to L$).
4. Aggregate probabilities using confidence-weighted pooling:
$$
P_{\text{TTA}}(x) = \sum_{k \in \{L, C, R\}} \frac{\alpha_k}{\sum \alpha_j} P_k(x), \quad \alpha_k = \max_c P_k(y=c|x)
$$
This automatically suppresses crops that capture only silent margins.

### Two-Stage Cascaded OOD Rejection Engine
* **Stage 1 (Global Unknown Filter):** Multi-embedding Cosine kNN-5 distance combined with ensemble entropy. Reject as Unknown if OOD score exceeds $\tau_{\text{global}}$ (calibrated to target 33% to 37% test Unknown ratio).
* **Stage 2 (Adaptive Prototype Shrinkage for Cendrawasih):** Samples provisionally assigned to Cendrawasih must satisfy a strict core radius:
$$
\text{Dist}_{\text{kNN}}(x, \text{Cendrawasih Centroid}) \le \tau_{\text{cendra, shrink}}
$$
where $\tau_{\text{cendra, shrink}}$ is set at the 84th percentile of in-distribution Cendrawasih OOF distance. Peripheral samples are redirected to Unknown.

---

## 8. Day 2 Submission Quota Allocation Strategy (5 Slots)

| Slot | Target Submission | Core Hypothesis & Method | Target Public LB |
|---|---|---|---|
| **Sub 1 (sub_005)** | `experiment 5.csv` | **Quick-Win Intermediate Baseline:** SNR-Dynamic Gated Blending + Two-Stage Prototype Shrinkage on Day 1 artifacts (zero retraining, CPU < 2 min). Establishes report ablation proof for Dynamic Gating. | **0.6050 - 0.6200** |
| **Sub 2 (sub_006)** | `experiment 6.csv` | **3-Crop TTA on Day 1 Models:** Applies 3-crop multi-window pooling to all 4 DL backbones. Tests recovery of truncated peripheral calls on audio >10s. | **0.6250 - 0.6400** |
| **Sub 3 (sub_007)** | `experiment 7.csv` | **Dual-Tree Upgrade:** 50% LightGBM + 50% XGBoost trained on new 818-dim feature matrix with VAD and high-harmonic ratios. | **0.6400 - 0.6600** |
| **Sub 4 (sub_008)** | `experiment 8.csv` | **Retrained ConvNeXt-Tiny v2.0:** ConvNeXt trained with Energy-Biased Jittered VAD and spectral subtraction. | **0.6650 - 0.6900** |
| **Sub 5 (sub_009)** | `experiment 9.csv` | **Master Multi-View Blend Day 2:** Full ensemble combining Preprocessing v1.0 and v2.0 models + 3-Crop TTA + Dynamic Gated Blending + Cascaded OOD Rejection. | **0.6900 - 0.7200** |

---

## 9. Technical Report Integration Plan (65% Weight)

Each experiment directly populates specific rubric requirements in the 10-page final report:

* **Bab Preprocessing & Feature Engineering (20%):**
  * Section *"Why Energy-Biased Jittered VAD Over Blind Center-Crop"* (includes waveform energy plots).
  * Section *"Why Half-Wave Rectified Spectral Subtraction Over Raw Z-Score"* (includes comparative spectrogram figure).
  * Section *"Acoustic Justification of Ultrasonic Harmonic Ratios (8-16 kHz)"*.
* **Bab Modeling (25%):**
  * Multi-family comparison table across 5 architectures.
  * Formal derivation of Class-Balanced Focal Loss ($\gamma=2.0, \beta=0.99$).
  * Formulation of SNR-Dynamic Gated Blending as an acoustic analogue to time-series progressive weighting.
* **Bab Evaluasi & Validasi (30%):**
  * Leakage-free 4-Fold StratifiedGroupKFold proof.
  * Comprehensive ablation table (ABL-01 through ABL-08).
  * OOD AUROC curves comparing MSP (0.9283), Energy (0.8209), and 256-d kNN (0.9970).
  * Confusion matrix progression across submission iterations.

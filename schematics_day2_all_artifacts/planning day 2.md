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


### 2.1 Tiga Temuan Emas dari Audit Rekan Tim (Jere EDA) yang Diadopsi

Audit terhadap notebook rekan tim (`notebook jere/notebook.ipynb`) menghasilkan 3 temuan empiris valid yang diadopsi ke dalam pipeline dan laporan kompetisi:

1. **Uji Efek Perancu / Confounder Effect Size (Bukti Formal untuk Bab Validasi):**
   * Uji non-parametrik Kruskal-Wallis pada Bondol Jawa membuktikan bahwa klaster perekam (`source_group`) mengendalikan **23.66% varians akustik** energi pita menengah ($H = 270.32, p = 1.855 \times 10^{-55}, \eta^2 = 0.2366$).
   * Pada Kutilang, `source_group` menyumbang **3.96% varians** ($H = 27.36, p = 4.942 \times 10^{-6}, \eta^2 = 0.0396$).
   * *Adopsi:* Menjadi justifikasi matematis tak terbantahkan di Bab Validasi bahwa partisi acak (*random K-Fold*) akan membocorkan tanda tangan mikrofon, sehingga **4-Fold StratifiedGroupKFold mutlak diperlukan**.

2. **Uji Signifikansi Separabilitas Bioakustik Spektral (Bukti Formal untuk Bab EDA):**
   * *Mid-band energy ($1\text{--}4\text{ kHz}$):* $H = 455.45, p = 1.26 \times 10^{-99}$ (Kutilang dominan, median 0.6405).
   * *High-band energy ($>4\text{ kHz}$):* $H = 391.45, p = 9.94 \times 10^{-86}$ (Bondol Jawa dominan, median 0.4196).
   * *Dominant Frequency:* $H = 191.14, p = 3.12 \times 10^{-42}$ (Cendrawasih paling rendah, median 1,992.19 Hz).
   * *Adopsi:* Nilai uji $H$ dan $p$-value ini langsung dicantumkan pada tabel karakteristik bioakustik Bab EDA laporan untuk membuktikan separabilitas spesies sebelum pemodelan.

3. **Karakteristik Fisik Sampel Anomali Unknown (Dasar Desain Filter Sinyal):**
   * Sampel uji yang menyimpang dari manifold latih memiliki frekuensi dominan yang anjlok ke median **1,359.38 Hz** (vs in-distribution 2,929.69 Hz) dan energi derau frekuensi rendah ($<1\text{ kHz}$) yang melonjak hampir 2x lipat menjadi **19.33%** (vs in-distribution 10.64%).
   * *Adopsi:* Menjadi dasar fisik pemasangan filter **High-Pass 300 Hz** dan **Stationary Spectral Subtraction** untuk merontokkan derau lingkungan $<1\text{ kHz}$ agar tidak disalahartikan sebagai Cendrawasih.
   * *Filter Kritis (Penolakan):* Kita **menolak keras** ambang batas Unknown 14.25% milik Jere. Estimasi tersebut runtuh di LB (0.210) karena 8 fitur skalar global tanpa resolusi temporal meremehkan populasi Unknown sebenarnya (~35%).

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

## 4. Fase 2: Upgraded Tabular Feature Matrix (~818 Dims) & Triad-Tree Ensemble

### Feature Matrix Enhancements (from 792 to ~818 features)
1. **Three High-Harmonic Bimodal Ratios (Capturing Cendrawasih up to 16 kHz):**
   * `ratio_low_to_mid`: $E_{0.3\text{--}2\text{kHz}} / (E_{2\text{--}6\text{kHz}} + \epsilon)$ (high for Cendrawasih, low for Kutilang).
   * `ratio_high_harmonic`: $E_{8\text{--}16\text{kHz}} / (E_{0.3\text{--}8\text{kHz}} + \epsilon)$ (differentiates Cendrawasih 9.08% harmonic energy from low-frequency wind rumble).
   * `bimodal_product`: $\sqrt{E_{0.3\text{--}2\text{kHz}} \times E_{8\text{--}16\text{kHz}}} / (E_{\text{total}} + \epsilon)$ (peaks exclusively when low rumble and ultrasonic overtones co-occur).
2. **Harmonic-Temporal Rhythm Index:** Autocorrelation of dominant mel bins across lags of 0.1s to 0.8s, separating rhythmic biological chirps from continuous abiotic noise (rain, wind).
3. **Distortion Descriptors:** `clipping_ratio` and `crest_factor`.

### Triad-Tree Head-to-Head & Blending
* **Model 1: LightGBM (Track A1):** Leaf-wise tree growth, `class_weight='balanced'`, 1000 estimators, learning rate 0.03.
* **Model 2: XGBoost (Track A2):** Depth-wise tree growth (`max_depth=6`), GPU histogram tree method, balanced sample weights.
* **Model 3: CatBoost (Track A3):** Symmetric oblivious trees (depth=6), GPU acceleration (`task_type='GPU'`), balanced sample weights. Oblivious trees provide maximum invariance against overfitting on high-dimensional noisy continuous features.
* **Ensemble Strategy:** Rather than discarding any model, combine LightGBM (asymmetric leaf-wise) + XGBoost (level-wise) + CatBoost (symmetric oblivious) probabilities. Blending three distinct decision boundary paradigms minimizes prediction variance on the extreme minority class (Cendrawasih).

---

## 5. Fase 3: Deep Learning Retraining with Preprocessing v2.0

### Model Training Focus
1. **Track B2 (ConvNeXt-Tiny, ~28.0M params):** Retrained with Energy-Biased Jittered VAD input and spectral subtraction. Remains the primary backbone due to its 0.8316 Cendrawasih capability.
2. **Track B4 (Swin-Tiny Transformer, ~28.3M params):** Retrained on cleaned spectrograms. Shifted-window self-attention operates with higher fidelity once stationary noise streaks are eliminated.
3. **Cross-Preprocessing Multi-View Retention:** Retain checkpoints and OOF probabilities of Day 1 **SE-ResNeXt-50 (Track B3)** and **EfficientNet-B0 (Track B1)**. Ensembling models trained on raw Log-Mel (v1.0) with models trained on spectrally subtracted Log-Mel (v2.0) introduces structural data-view diversity that protects against spectral subtraction artifacts.

---

## 6. Fase 4: Acoustic Noise-Conditioned Dynamic Gating (Ekuivalen Progressive Weighting)

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

## 8. Day 2 Submission Quota Allocation Strategy & Ablation Design

Untuk membuktikan keunggulan metodologi pada laporan teknis (bobot 65%), Day 2 dirancang menghasilkan dua pengujian terisolasi yang memisahkan dampak Ide 1 (SNR-Dynamic Gating) dan Ide 2 (Adaptive Prototype Shrinkage):

| Slot | Target Submission | Core Hypothesis & Method | Target Public LB | Nilai Ilmiah untuk Laporan (Bab Validasi 30%) |
|---|---|---|---|---|
| **Sub 1 (sub_005)** | `experiment 5.csv` | **Acoustic Noise-Conditioned Dynamic Gating (Ide 1 Murni):** Menginterpolasi bobot Tabular Triad dan Deep Learning v2.0 secara dinamis berbasis SNR Proxy. Menggunakan ambang batas penolakan global standar $\tau_{\text{global}}$ ($p91.5$). | **0.6300 - 0.6600** | **Isolasi Dampak Ide 1:** Membuktikan bahwa alokasi bobot adaptif berbasis noise floor mengungguli rata-rata statis unweighted NLL. |
| **Sub 2 (sub_006)** | `experiment 6.csv` | **Dynamic Gating + Adaptive Prototype Shrinkage + 3-Crop TTA (Ide 1 + Ide 2):** Menambahkan gerbang tahap kedua khusus Cendrawasih ($p84$ core radius) dan inferensi multi-window TTA untuk memulihkan vokal di tepi rekaman. | **0.6800 - 0.7200** | **Isolasi Dampak Ide 2:** Membuktikan eliminasi 100% false positive Cendrawasih akibat desis frekuensi rendah di data uji. |
| **Sub 3-5** | Cadangan | Alokasi kalibrasi lanjutan atau fine-tuning ambang batas jika diperlukan. | **0.7000+** | Stabilitas estimasi sebaran Unknown. |

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

## 10. Rencana Eksekusi Lengkap Cell-by-Cell Notebook Day 2 (End-to-End Master)

Notebook master Day 2 dirancang berjalan secara mandiri dan utuh dari nol (Cell 1) hingga submission final (Cell 26). Struktur ini mengintegrasikan seluruh tahapan EDA deep-dive Day 1, penambahan uji formal Kruskal-Wallis & Confounder Effect Size (Jere), Preprocessing v2.0 (VAD + Spectral Subtraction + Active RMS), Dual-Tree Tabular (LightGBM + XGBoost), Retraining Deep Learning (ConvNeXt-Tiny + Swin-Tiny), Multi-View Ensembling, Dynamic Gating, 3-Crop TTA, hingga Two-Stage Cascaded OOD.

---

### Bagian I: Fondasi Lingkungan, Metadata, & Deep Bioacoustic EDA (Cell 1 s.d. Cell 8)

* **Cell 1: Environment Setup, Deterministic Seeds, & Hardware Config**
  * *Operasi:* Impor dependensi (`torch`, `torchaudio`, `timm`, `librosa`, `soundfile`, `lightgbm`, `xgboost`, `scipy`, `sklearn`), penentuan seed deterministik tunggal (42) pada PyTorch, CuDNN benchmark/deterministic, NumPy, dan Python stdlib.
  * *Jalur:* Autodeteksi lingkungan eksekusi (Kaggle `/kaggle/input/competitions/schematics-2026` vs lokal `C:/Github/schematics-npc-dsc-2026`).
  * *Output:* Konfirmasi alokasi GPU (`cuda:0`, nama perangkat, VRAM bebas) dan direktori kerja.

* **Cell 2: Metadata Schema Audit & Target Distribution**
  * *Operasi:* Membaca `metadata.csv` (1,826 baris train) dan `sample_submission.csv` (814 baris test).
  * *Audit Data:* Verifikasi 0 missing values, cross-tabulation 3 spesies train (Bondol Jawa: 1,124 / 61.56%, Kutilang: 619 / 33.90%, Cendrawasih: 83 / 4.55%), menghitung Imbalance Ratio ($IR = 13.54$).
  * *Output:* Log distribusi kelas dan rasio perbandingan kelas.

* **Cell 3: Source Group Nesting, Duration KS-Test, & Deduplication**
  * *Operasi:* Verifikasi nesting 15 `source_group` terhadap label (100% nested per spesies: 7 Bondol, 4 Kutilang, 4 Cendrawasih). Uji Kolmogorov-Smirnov pada durasi train vs test (KS stat 0.0302, $p = 0.6695$, membuktikan zero temporal domain shift).
  * *Deduplikasi:* Audit hash SHA256 waveform (1,826 train unik 100%, 814 test unik 100%, 0 overlap train-test).
  * *Output:* Konfirmasi ketiadaan duplikasi dan kesetaraan distribusi durasi (~10.04s).

* **Cell 4: Acoustic Health & Hardware Distortion Audit (Clipping & Dynamic Range)**
  * *Operasi:* Mendeteksi saturasi sensor mikrofon lapangan (persentase rekaman dengan peak amplitudo $\ge 0.99$). Menghitung ZCR rata-rata per spesies dan rasio keheningan (*silence ratio* pada threshold $-40\text{ dBFS}$).
  * *Temuan:* 40% s.d. 55% file train dan test mengalami digital clipping (peak hingga 1.004). ZCR Kutilang 0.2899 (tonal), Bondol 0.3483, Cendrawasih 0.3699.
  * *Output:* Rangkuman tabel diagnostik kesehatan sinyal mentah.

* **Cell 5: Power Spectral Density (Welch PSD) & Uji Separabilitas Bioakustik Formal**
  * *Operasi:* Menghitung Welch PSD (nperseg=2048) pada rekaman mono. Membagi spektrum menjadi 4 pita frekuensi: Sub-low ($0\text{--}2\text{ kHz}$), Mid-low ($2\text{--}4\text{ kHz}$), Mid-high ($4\text{--}6\text{ kHz}$), dan High ($6\text{--}8\text{ kHz}$).
  * *Integrasi Temuan Jere:* Uji non-parametrik Kruskal-Wallis antar spesies pada Mid-band ($H = 455.45, p = 1.26 \times 10^{-99}$), High-band ($H = 391.45, p = 9.94 \times 10^{-86}$), dan Dominant Frequency ($H = 191.14, p = 3.12 \times 10^{-42}$).
  * *Output:* Bukti matematis pembeda spektral tajam: Kutilang dominan di 2-4 kHz (57.2%), Cendrawasih bimodal di 0-2 kHz (44.0%) dan 6-8 kHz (14.7%).

* **Cell 6: Audit Efek Perancu Source Group (Kruskal-Wallis Eta-Squared)**
  * *Operasi:* Menghitung besaran varians energi akustik yang terikat pada klaster sesi perekaman di dalam spesies yang sama (formula $\eta^2 = \max(0, (H - k + 1) / (n - k))$).
  * *Bukti Confounder:* Bondol Jawa memiliki $H = 270.32, p = 1.855 \times 10^{-55}, \mathbf{\eta^2 = 0.2366}$ (23.66% varians akustik murni akibat mikrofon/sesi perekam!). Kutilang $\eta^2 = 0.0396$.
  * *Output:* Justifikasi formal laporan teknis mengapa K-Fold biasa haram dan Stratified Group K-Fold mutlak wajib.

* **Cell 7: Visualisasi Publikasi Komprehensif (3-Panel EDA)**
  * *Operasi:* Membuat gambar komparatif visualisasi publication-grade: Panel 1 (Distribusi kelas & struktur nesting source group), Panel 2 (Kurva rata-rata Welch PSD 4 pita per spesies), Panel 3 (Scatter separasi akustik 2D: ZCR vs Spectral Centroid).
  * *Output:* Menyimpan gambar beresolusi tinggi `eda_acoustic_profiles.png` untuk Bab EDA laporan.

* **Cell 8: Pre-Processing Audit (High-Frequency Energy, Stereo Correlation, & Mel-Spectrograms)**
  * *Operasi:* Audit spektral frekuensi tinggi 48 kHz native (membuktikan energi $>8\text{ kHz}$ hanya $<9.3\%$, pita $16\text{--}24\text{ kHz}$ hanya $<1.22\%$, sehingga downsample 32 kHz aman mutlak). Audit korelasi kanal stereo (0.724 s.d. 0.800 membuktikan mono downmix lossless).
  * *Visualisasi:* Plot mel-spektrogram representatif per kelas beserta audio kandidat Unknown.
  * *Output:* Menyimpan gambar `eda_mel_spectrograms.png`.

---

### Bagian II: Preprocessing v2.0 & Partisi Bebas Leakage (Cell 9 s.d. Cell 11)

* **Cell 9: Penguncian Partisi 4-Fold StratifiedGroupKFold on `source_group`**
  * *Operasi:* Menjalankan partisi 4-Fold `StratifiedGroupKFold` pada 1,826 audio train dengan target stratifikasi kelas dan pengelompokan `source_group`.
  * *Verifikasi Nol Kebocoran:* Alokasi 4 grup Cendrawasih tepat 1 grup per fold (Fold 0: 35 sampel, Fold 1: 29 sampel, Fold 2: 16 sampel, Fold 3: 3 sampel). Zero group overlap antar fold.
  * *Output:* Menyimpan `metadata_with_folds.csv` dan mengunci parameter ke `preprocess_config.json`.

* **Cell 10: Audio Preprocessing Engine v2.0 (The Audio DSP Core)**
  * *Operasi:* Membangun modul pemrosesan sinyal v2.0:
    1. `resample_poly_32k`: Anti-aliasing polyphase FIR filter (anti-Gibbs ringing pada sinyal clipping).
    2. `butter_highpass_300hz`: Butterworth filter orde-4 memotong gemuruh angin $<300\text{ Hz}$.
    3. `active_frame_rms_norm`: Normalisasi energi frame aktif ($>-40\text{ dBFS}$) dengan kompresi soft $\tanh$ limiter.
    4. `compute_vad_window_indices`: Sliding window 10.0s berbasis densitas energi vokal $2\text{--}8\text{ kHz}$ (mode deterministik untuk tabular/validasi; mode jittered $\pm 1.5\text{s}$ untuk training DL).
    5. `stationary_spectral_subtraction`: Pengurangan noise floor kuantil ke-10 dengan decibel half-wave rectification $\max(S_{\text{dB}} - P_{10}, 0.0)$.
  * *Output:* Modul fungsi pipeline sinyal audio terverifikasi.

* **Cell 11: Unit Verification & Visualisasi Preprocessing v2.0**
  * *Operasi:* Eksekusi pipeline v2.0 pada 3 sampel uji (Bondol bersih, Cendrawasih ter-clipping, dan Test bernoise).
  * *Pemeriksaan:* Dimensi tensor spektrogram tepat `(1, 128, 626)`, Z-score rata-rata 0.000, standar deviasi 1.000, nilai minimum $\ge 0.0$, nol NaN/inf.
  * *Output:* Menyimpan plot perbandingan visual `eda_day2_preprocessing_v2_verification.png`.

---

### Bagian III: Feature Engineering & Triad-Tree Tabular Engine (Cell 12 s.d. Cell 16)

* **Cell 12: Ekstraksi Fitur Tabular Bioakustik v2.0 (~818 Dimensi)**
  * *Operasi:* Ekstraksi matriks fitur kaya informasi bioakustik pada 1,826 train dan 814 test audio:
    - 512 fitur: Statistik Log-Mel 128 bin (mean, std, p10, p90).
    - 256 fitur: Delta dan Delta-2 spectral dynamics (mean, std).
    - 24 fitur: Deskriptor spektral (Centroid, Bandwidth, Rolloff, Flatness, ZCR, RMS - masing-masing 4 statistik).
    - 12 fitur: Tiga Rasio Bimodal & Harmonik Tinggi Cendrawasih ($0.3\text{--}2\text{ kHz}$ vs $2\text{--}6\text{ kHz}$, $8\text{--}16\text{ kHz}$ vs $0.3\text{--}8\text{ kHz}$, dan produk bimodal).
    - 8 fitur: Autokorelasi temporal ritme pulsa pada 4 mel-bin dominan (lag 0.1s s.d. 0.8s).
    - 6 fitur: Hardware distortion descriptors (`clipping_ratio`, `crest_factor`, active-to-total ratio).
  * *Output:* Menyimpan file cache `track_a_X_train_32k_v2.npy` (1826, 818) dan `track_a_X_test_32k_v2.npy` (814, 818).

* **Cell 13: Track A1 - LightGBM GBDT Training (4-Fold)**
  * *Operasi:* Melatih 4 model LightGBM pada partisi 4-Fold StratifiedGroupKFold (`class_weight='balanced'`, `n_estimators=1000`, `learning_rate=0.03`, early stopping 50).
  * *Output:* Menyimpan `oof_track_a1_lgbm_v2.csv`, `oof_p_lgbm_v2.npy`, probabilitas test `test_p_lgbm_v2.npy`, dan mencetak Global OOF Macro F1 serta F1 per kelas.

* **Cell 14: Track A2 - XGBoost GPU Training (4-Fold)**
  * *Operasi:* Melatih 4 model XGBoost dengan metode histogram GPU (`tree_method='hist'`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `learning_rate=0.03`, sample weight seimbang).
  * *Output:* Menyimpan `oof_track_a2_xgboost_v2.csv`, `oof_p_xgb_v2.npy`, probabilitas test `test_p_xgb_v2.npy`, dan mencetak Global OOF Macro F1 serta F1 per kelas.

* **Cell 15: Track A3 - CatBoost GPU Training (4-Fold)**
  * *Operasi:* Melatih 4 model CatBoost dengan pohon simetris (*oblivious trees*, depth 6) pada GPU (`task_type='GPU'`, `learning_rate=0.04`, sample weight seimbang).
  * *Output:* Menyimpan `oof_track_a3_catboost_v2.csv`, `oof_p_catboost_v2.npy`, probabilitas test `test_p_catboost_v2.npy`, dan mencetak Global OOF Macro F1 serta F1 per kelas.

* **Cell 16: Triad-Tree Tabular Evaluation & Optimal Blend**
  * *Operasi:* Evaluasi komparatif 3 paradigma pohon: LightGBM (leaf-wise), XGBoost (depth-wise), dan CatBoost (symmetric oblivious). Mengoptimalkan bobot gabungan tabular via Macro F1 Nelder-Mead atau weighted average.
  * *Output:* Matriks probabilitas gabungan tabular `oof_p_tabular_v2.npy` dan `test_p_tabular_v2.npy`.

---

### Bagian IV: Deep Learning Retraining on Cleaned Manifolds (Cell 17 s.d. Cell 21)

* **Cell 17: Deep Learning GPU Frontend & BirdChirpNet v2 Architecture**
  * *Operasi:* Modul PyTorch STFT GPU frontend terintegrasi dengan spectral subtraction. Kelas arsitektur `BirdChirpNetV2` (mendukung ConvNeXt-Tiny, Swin-Tiny, SE-ResNeXt-50, EffNet-B0, GeM pooling adaptif $p=3.0$, bottleneck head 256-d L2-normalized).
  * *Loss Function:* `ClassBalancedFocalLoss` ($\gamma=2.0, eta=0.99, 	ext{smoothing}=0.05$).
  * *Output:* Inisialisasi model siap latih dan verifikasi forward pass tensor audio `(32, 320000)`.

* **Cell 18: Track B2 Retraining - ConvNeXt-Tiny with Jittered VAD (4-Fold)**
  * *Operasi:* Melatih ulang ConvNeXt-Tiny (~28.0M params) dengan protokol LP-FT 2-phase (5 epoch Linear Probing + 18 epoch Fine-Tuning) menggunakan `Energy-Biased Jittered Cropping` ($\pm 1.5	ext{s}$). Waveform di-cache di RAM.
  * *Output:* Menyimpan model weights di `checkpoints_b2_v2_convnext_tiny/`, probabilitas OOF `oof_b2_v2_probs.npy` (Global OOF Macro F1 = 0.9323, Cendrawasih = 0.9448), embedding OOF `oof_b2_v2_emb256d.npy`, dan reference train embeddings `train_fold_b2_v2_emb256d.pt`.

* **Cell 19: Track B4 Retraining - Swin-Tiny Shifted Window Attention (4-Fold)**
  * *Operasi:* Melatih ulang Swin-Tiny (~28.3M params) pada spektrogram hasil Preprocessing v2.0 menggunakan protokol LP-FT 2-phase.
  * *Output:* Menyimpan model weights di `checkpoints_b4_v2_swin_tiny/`, probabilitas OOF `oof_b4_v2_probs.npy` (Global OOF Macro F1 = 0.9098, Cendrawasih = 0.8671), dan embedding OOF `oof_b4_v2_emb256d.npy`.

* **Cell 20: Track B3 Retraining - SE-ResNeXt-50 32x4d with Channel Attention (4-Fold)**
  * *Operasi:* Melatih ulang SE-ResNeXt-50 32x4d (`legacy_seresnext50_32x4d.in1k`, ~27.6M params) dengan mekanisme *Squeeze-and-Excitation channel attention* pada spektrogram Preprocessing v2.0 menggunakan protokol LP-FT 2-phase.
  * *Output:* Menyimpan model weights di `checkpoints_b3_v2_se_resnext50/`, probabilitas OOF `oof_b3_v2_probs.npy`, embedding OOF `oof_b3_v2_emb256d.npy`, dan inferensi test set 1-crop `test_b3_v2_probs_1crop.npy`.

* **Cell 21: Track B1 Retraining - EfficientNet-B0 Depthwise Convolutions (4-Fold)**
  * *Operasi:* Melatih ulang EfficientNet-B0 (~5.3M params) dengan konvolusi terpisah kedalaman (*depthwise separable convolutions*) pada audio Preprocessing v2.0 menggunakan protokol LP-FT 2-phase.
  * *Output:* Menyimpan model weights di `checkpoints_b1_v2_effnet_b0/`, probabilitas OOF `oof_b1_v2_probs.npy`, embedding OOF `oof_b1_v2_emb256d.npy`, dan inferensi test set 1-crop `test_b1_v2_probs_1crop.npy`.

---

### Bagian V: Model Grand Prix, Dynamic Gating, TTA, & Dual-Submission (Cell 22 s.d. Cell 26)

* **Cell 22: Comprehensive Model Grand Prix & Decision Matrix (Bandingkan Semua Model)**
  * *Operasi:*
    1. Menyusun satu tabel perbandingan komprehensif seluruh model yang telah dilatih pada data partisi yang sama:
       - 3 Model Tabular Standalone: LightGBM (0.8475), XGBoost (0.8408), CatBoost (0.8322).
       - Tabular Dual-Tree Blend: 50% LightGBM + 50% XGBoost (0.8486).
       - 4 Model Deep Learning Retrained v2.0: ConvNeXt-Tiny (0.9323), Swin-Tiny (0.9098), SE-ResNeXt-50, dan EfficientNet-B0.
    2. Menghitung matriks korelasi prediksi probabilitas antar seluruh arsitektur untuk melihat derajat redundansi vs ortogonalitas.
    3. Evaluasi performa per kelas (terutama Cendrawasih F1 dan False Positive Rate).
  * *Sesi Brainstorming & Keputusan:* Memilih sub-portofolio model paling defensible dan efisien untuk diintegrasikan ke final pipeline (mengeliminasi model redundan agar penjelasan laporan ringkas dan berbobot).
  * *Output:* Tabel komparasi lengkap dan artefak `model_grand_prix_summary.json`.

* **Cell 23: Acoustic Noise-Conditioned Dynamic Gating (Ide 1 - Ekuivalen Progressive Weighting)**
  * *Premis Fisik:*
    - Model Tabular Triad mengandalkan statistik agregat global; sangat presisi pada rekaman jernih (High-SNR, flatness rendah), namun terdistorsi oleh derau hujan/angin kontinu.
    - Model Deep Learning (ConvNeXt, Swin, SE-ResNeXt, EffNet) melihat spektrogram sebagai citra 2D spasial; sangat tangguh menyaring garis vokal burung meskipun tertutup desis latar belakang (noise-invariant spatial filter).
  * *Mekanisme Interpolasi Dinamis Kontinu:*
    Menghitung $\text{SNR}_{\text{proxy}}(x_i) \in [0, 1]$ dari Spectral Flatness dan active RMS.
    $$w_{\text{Tab}}(x_i) = w_{\min} + (w_{\max} - w_{\min}) \cdot \left(\text{SNR}_{\text{proxy}}(x_i)\right)^\gamma$$
    $$w_{\text{DL}}(x_i) = 1.0 - w_{\text{Tab}}(x_i)$$
    - Audio jernih ($\text{SNR}_{\text{proxy}} \to 1$): $w_{\text{Tab}}$ naik hingga 0.50 (memanfaatkan presisi tinggi fitur fisik).
    - Audio berderau lebat ($\text{SNR}_{\text{proxy}} \to 0$): $w_{\text{DL}}$ mendominasi hingga 0.85 - 0.90 (didominasi ConvNeXt-Tiny dan Swin-Tiny).
  * *Output:* Matriks probabilitas OOF tertutup dinamis `oof_p_gated_master.npy` dan test set `test_p_gated_master.npy`.

* **Cell 24: 3-Crop Test-Time Augmentation (TTA) on Test Set**
  * *Operasi:* Inferensi multi-window 10.0s (Left $0\text{--}10\text{s}$, Center, Right $(L-10)\text{--}L$) pada model-model Deep Learning terpilih untuk 814 rekaman data uji.
  * *Agregasi:* Confidence-weighted softmax pooling untuk probabilitas kelas dan L2-normalized mean pooling untuk vektor embedding 256-d.
  * *Output:* Matriks probabilitas uji TTA teragregasi dan matriks embedding uji TTA per fold.

* **Cell 25: Two-Stage Cascaded OOD Rejection Engine**
  * *Operasi:*
    - *Stage 1 (Global OOD Filter):* Menghitung jarak Cosine kNN-5 pada embedding ConvNeXt-Tiny v2.0 256-d, digabung dengan Gated Softmax Entropy. Menetapkan ambang $\tau_{\text{global}}$ untuk menangkap ~34% s.d. 37% Unknown.
    - *Stage 2 (Adaptive Prototype Shrinkage Cendrawasih):* Mengunci radius penerimaan kelas Cendrawasih pada persentil ke-84 data training. Calon Cendrawasih yang berada di luar radius dialihkan ke Unknown.
  * *Output:* Audit sebaran prediksi 4 kelas dan verifikasi jumlah Unknown.

* **Cell 26: Dual-Submission Assembly, Verification, & Bundling (sub_005 & sub_006)**
  * *Operasi:* Menyusun dua berkas submission resmi:
    1. `experiment 5.csv` (`sub_005`): Menilai dampak murni Dynamic Gating (Ide 1) dengan ambang global.
    2. `experiment 6.csv` (`sub_006`): Menilai dampak sinergi Dynamic Gating + Adaptive Prototype Shrinkage Cendrawasih (Ide 1 + Ide 2) + 3-Crop TTA.
  * *Pemeriksaan Integritas Ketat pada Kedua Berkas:*
    1. Tepat 814 baris.
    2. Kolom persis `file_id` dan `label`.
    3. `file_id` cocok 100% dengan `sample_submission.csv`.
    4. Bebas NaN / null.
    5. Label hanya terdiri dari 4 kelas resmi: `Bondol Jawa`, `Cendrawasih`, `Kutilang`, `Unknown`.
  * *Penyimpanan Bundel:* Menyimpan `submission.csv`, `metrics.json`, dan notebook ke folder `submissions/sub_005_<timestamp>/` dan `submissions/sub_006_<timestamp>/`.
  * *Output:* Tautan unduh interaktif `FileLink("experiment 5.csv")` dan `FileLink("experiment 6.csv")`.

# MASTER RESEARCH BLUEPRINT v3.0 (`final planning.md`)
**Judul Proyek:** End-to-End Bioacoustic Open-Set Recognition via Transductive Foundation Model Alignment and Orthogonal Forensic Shielding  
**Target Artefak:** Notebook Komputasi (`notebooks/schematics_master_pipeline.ipynb`) & Naskah Laporan Teknis 10 Halaman  
**Standar:** Zero Data Leakage · Full Retrain from Raw FLAC · Zero LB-Probing Quota · Inline Visualization  
**Total Sel:** 31 | **Estimasi Runtime:** ~5–6 jam (T4 GPU)


## Peta 10 Fase × 31 Sel

| Fase | Sel | Fokus | Figur Dihasilkan |
| :--- | :---: | :--- | :--- |
| **1. Setup** | 1 | Environment, VRAM, determinisme | - |
| **2. EDA** | 2–7 | Dataset, distribusi, sinyal, spektral, confounder, fold | 6 figur |
| **3. Preprocessing & FE** | 8–10 | DSP engine, caching, 818-d fitur, SHAP | 3 figur |
| **4. Model Grand Prix** | 11–15 | GBDT, CNN, AST, summary, ROC/PR | 4 figur |
| **5. Foundation Models** | 16–19 | Perch v2, Multi-Taxa, BirdNET, Hybrid CosineHead | 3 figur |
| **6. Forensic Shield** | 20–21 | Threshold BirdNET, Dual Shield | 2 figur |
| **7. Self-Training** | 22–24 | Curation, R1+R2, R3 saturation proof | 2 figur |
| **8. Architecture Optim.** | 25–28 | Head ablation, augmentation, ensemble, threshold sweep | 4 figur |
| **9. Grand Synthesis** | 29–30 | Optimal config, final submission export | 1 figur |
| **10. Delivery** | 31 | Ablation table, archiving, hash | 1 figur |
| | | | **~26 figur total** |


## FASE 1: Environment

### Cell 1: Environment Setup, VRAM Allocation & Determinisme
* **Prosedur:**
  1. Variabel lingkungan: `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, `TF_CPP_MIN_LOG_LEVEL="2"`, `PYTHONHASHSEED="42"`.
  2. `tf.config.experimental.set_memory_growth(gpu, True)` sebelum PyTorch init.
  3. Seed global $S=42$: `random`, `numpy`, `torch`, `tf`. Aktifkan `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.benchmark=False`.
  4. Helper: `purge_vram()`, `seed_worker()`, `make_generator()`.
  5. Helper visualisasi: `setup_plot_style()` (DPI 150, sans-serif, tight_layout), `save_fig(fig, name)`.
  6. **Tanpa saklar cache.** Seluruh pipeline dari raw `.flac`.
* **Output:** Tabel versi library + status GPU di stdout.


## FASE 2: Comprehensive Bioacoustic EDA

### Cell 2: Dataset Discovery, Schema & Class Distribution
* **Tujuan:** Verifikasi integritas data dan kuantifikasi ketimpangan kelas.
* **Prosedur:**
  1. Verifikasi skema metadata latih ($N_{\text{train}} = 1.826$) dan uji ($N_{\text{test}} = 814$).
  2. Hitung $IR = \max(n_c)/\min(n_c) = 13.54$, $H' = -\sum p_c \ln p_c = 0.832$.
  3. **Figur `fig_eda_class_distribution`** (3-panel, 12×4):
     * Panel A: Horizontal bar chart + annotasi $n_c$ dan $p_c\%$.
     * Panel B: Donut chart dengan $N_{\text{total}}$ di pusat.
     * Panel C: Effective number $E_n = (1-\beta^n)/(1-\beta)$ untuk $\beta \in \{0.9, 0.99, 0.999\}$.
* **Keputusan:** Class-Balanced Focal Loss ($\beta=0.99, \gamma=2.0$) wajib pada semua arsitektur DL.

### Cell 3: Audio Properties, Duration Shift Test & Clipping Forensics
* **Tujuan:** Properti fisik sinyal, invarian durasi, dan integritas amplitudo.
* **Prosedur:**
  1. Binary FLAC header parser → sampling rate, bit depth, channel, duration.
  2. KS-test durasi train vs test: $D_{\text{KS}} = 0.0302$, $p = 0.6695$.
  3. Audit clipping rate ($|x(t)| \ge 0.99$) dan korelasi stereo $r_{L,R}$.
  4. **Figur `fig_eda_signal_audit`** (2×2, 14×8):
     * Panel A: Overlapping histogram + KDE durasi train vs test.
     * Panel B: Violin plot durasi per kelas.
     * Panel C: Boxplot clipping rate per kelas.
     * Panel D: Scatter $r_{L,R}$ vs $\Delta_{\text{RMS}}$, diwarnai per kelas.
* **Keputusan:** Peak normalization dilarang → Active-RMS. Downmix mono $(ch_0+ch_1)/2$ aman.

### Cell 4: Waveform Gallery & Mel-Spectrogram Comparison
* **Tujuan:** Representasi visual kualitatif per spesies untuk Bab EDA laporan.
* **Prosedur:**
  1. Pilih 1 sampel representatif per kelas (median RMS).
  2. Hitung 128-band Log-Mel Spectrogram (32 kHz, $N_{\text{FFT}}=2048$, hop=512).
  3. **Figur `fig_eda_waveform_spectrogram`** (2×3, 16×8):
     * Baris 1: Waveform mono + VAD overlay per spesies.
     * Baris 2: Mel-spectrogram per spesies, shared colorbar. Garis putus-putus di pita frekuensi kunci.

### Cell 5: Welch PSD Spectral Decomposition & Nyquist Decision
* **Tujuan:** Bukti kuantitatif sampling rate minimum dan tanda akustik bimodal Cendrawasih.
* **Prosedur:**
  1. Welch PSD (Hann, $N_{\text{FFT}}=2048$, overlap 50%) pada seluruh 1.826 file.
  2. Integrasi fraksi daya 5 pita: 0–2, 2–4, 4–6, 6–8, 8–16 kHz.
  3. Kruskal-Wallis per pita.
  4. **Figur `fig_eda_spectral_energy`** (1×3, 16×4):
     * Panel A: PSD overlay (mean ± std shaded) per kelas, garis Nyquist di 8 kHz dan 16 kHz.
     * Panel B: 100% stacked bar proporsi energi per kelas per pita.
     * Panel C: Waterfall kehilangan energi saat downsample 48→32→16 kHz.
* **Keputusan:** 32 kHz default (melindungi 9.08% harmonik Cendrawasih). 48 kHz native untuk BirdNET.

### Cell 6: Confounder Audit & StratifiedGroupKFold Justification
* **Tujuan:** Pembuktian bias mikrofon dan justifikasi skema validasi.
* **Prosedur:**
  1. Kruskal-Wallis $H$ + $\eta^2 = (H-k+1)/(n-k)$ per kelas per fitur.
  2. **Figur `fig_eda_confounder`** (1×3, 16×4):
     * Panel A: Bar chart $\eta^2$ per kelas, garis ambang efek medium ($\eta^2=0.06$).
     * Panel B: Heatmap source_group × fitur spektral (Bondol Jawa).
     * Panel C: Perbandingan OOF F1 random-KFold vs GroupKFold (LightGBM cepat).
* **Keputusan:** StratifiedGroupKFold wajib. $\eta^2 = 0.2366$ pada Bondol Jawa.

### Cell 7: 4-Fold Combinatorial Generator & Fold Visualization
* **Tujuan:** Partisi validasi silang dengan isolasi 100% source_group.
* **Prosedur:**
  1. Pencarian kombinatorial: 15 grup → 4 fold, kendala 1 grup Cendrawasih per fold, $G_i \cap G_j = \emptyset$.
  2. Evaluasi wajib: **Global Pooled OOF Macro F1** (bukan rata-rata per-fold).
  3. **Figur `fig_eval_folds`** (1×2, 12×4):
     * Panel A: Stacked bar per fold (train/val) per kelas.
     * Panel B: Heatmap biner 15×4 (group × fold), highlight Cendrawasih.
* **Artefak:** `metadata_with_folds.csv`.


## FASE 3: Preprocessing, Feature Engineering & Interpretabilitas

### Cell 8: Preprocessing v2.0 & VAD-Guided Dual-Stream Caching
* **Tujuan:** DSP engine + caching seluruh dataset ke RAM dari raw FLAC.
* **Prosedur:**
  1. **Butterworth HPF Orde-4:** $f_c=300$ Hz (DSP/Vision), $f_c=100$ Hz (Foundation Model). `sosfiltfilt` zero-phase.
  2. **Active-RMS Normalization:** noise floor dari P10, frame aktif > +10 dB, target $-18$ dBFS.
  3. **Cumulative $O(N)$ VAD:** $E_{\text{cum}}[t]$ untuk top-3 jendela vokal non-redundan.
  4. **Stream A (32 kHz):** Polyphase FIR resample, 3 window × 5.0 s (160.000 sampel).
  5. **Stream B (48 kHz):** Native, 3 window × 3.0 s (144.000 sampel).
  6. **Figur `fig_preprocess_before_after`** (2×3, 14×8):
     * Baris 1: Raw waveform + mel-spectrogram (1 per kelas).
     * Baris 2: Processed + VAD markers + mel-spectrogram sesudah.
* **Output:** Train cache ~6.2 GB, Test cache ~2.8 GB, 0 silent windows.

### Cell 9: 818-d DSP Feature Extraction & Correlation Analysis
* **Tujuan:** Fitur tabular fisik + audit korelasi.
* **Prosedur:**
  1. 128-band Log-Mel statistik + $\Delta$ + $\Delta^2$ (512-d).
  2. Kontur spektral: centroid, bandwidth, roll-off, flatness, ZCR, RMS, 7-band contrast (256-d).
  3. Bioacoustic bimodal ratios (12-d) + temporal autocorrelation (14-d) + custom (24-d).
  4. **Figur `fig_preprocess_features`** (1×2, 14×6):
     * Panel A: Clustered correlation heatmap (hierarchical dendrogram).
     * Panel B: Ridge plot top-20 fitur varians tertinggi, dikondisikan per kelas.
* **Output:** $X_{\text{train}} \in \mathbb{R}^{1826 \times 818}$, $X_{\text{test}} \in \mathbb{R}^{814 \times 818}$.

### Cell 10: SHAP Feature Importance Analysis
* **Tujuan:** Interpretabilitas model-agnostik untuk Bab Preprocessing laporan.
* **Prosedur:**
  1. Latih 1-fold LightGBM cepat (fold 0, 200 trees).
  2. `shap.TreeExplainer` pada $\sim 1.323$ sampel latih fold 0.
  3. **Figur `fig_feature_shap`** (1×3, 16×10):
     * Panel A: SHAP beeswarm plot (top-25 fitur).
     * Panel B: Per-class mean $|\text{SHAP}|$ bar chart (top-15).
     * Panel C: Dependence plot untuk 2 fitur teratas.


## FASE 4: Model Grand Prix & Learning Dynamics

### Cell 11: Paradigma A - GBDT Grand Prix *(Full Retrain)*
* **Tujuan:** Baseline tabular: LightGBM, XGBoost, CatBoost secara simultan.
* **Prosedur:**
  1. 4-Fold StratifiedGroupKFold pada ketiga model, `compute_sample_weight('balanced', y)`.
  2. Simpan OOF probs (untuk ensemble dan SHAP).
  3. **Figur `fig_model_gbdt`** (1×2, 12×4):
     * Panel A: Normalized confusion matrix LightGBM (champion GBDT).
     * Panel B: Grouped bar per-fold per-class F1.
* **Hasil:** LGBM $0.8542$, CatBoost $0.8486$, XGBoost $0.8412$.

### Cell 12: Paradigma B - CNN Grand Prix: ConvNeXt, Swin, SE-ResNeXt *(Full Retrain)*
* **Tujuan:** Transfer learning 3 arsitektur vision pada Log-Mel Spectrogram 32 kHz.
* **Prosedur:**
  1. On-the-fly GPU Log-Mel Frontend (128 mel, $f_{\max}=16$ kHz).
  2. **LP-FT setiap model:** 5 epoch linear probing + 18 epoch full fine-tuning. Cosine Annealing LR + CB-Focal Loss.
  3. **Rekam per epoch:** train loss, val loss, val Macro F1, val Cendrawasih F1.
  4. Simpan OOF probs + best checkpoint per fold (untuk Exp-D ensemble nanti).
  5. **Figur `fig_model_cnn_learning`** (2×2, 14×10):
     * Panel A: Training loss overlay (3 model, mean±std 4 fold).
     * Panel B: Val Macro F1 overlay. Garis transisi LP→FT di epoch 5.
     * Panel C: Val Cendrawasih F1 overlay.
     * Panel D: Confusion matrix ConvNeXt-Tiny (champion CNN).
* **Hasil:** ConvNeXt $0.9323$, Swin $0.9098$, SE-ResNeXt $0.9005$.

### Cell 13: Paradigma C - Audio Transformer Investigation & Failure Proof *(Full Retrain)*
* **Tujuan:** Menguji Bird-MAE & BEATs dan membuktikan kegagalan arsitektur 16 kHz.
* **Prosedur:**
  1. Embedding 768-d + 4-Fold linear probing + partial fine-tune 8 epoch.
  2. Korelasikan FN Cendrawasih dengan rasio energi $8\text{–}16$ kHz.
  3. Inline: scatter plot FN vs high-freq energy ratio.
* **Hasil:** Bird-MAE $0.8715$, BEATs $0.8905$. Paradigma C **dieliminasi** - Nyquist 8 kHz membutakan Cendrawasih.

### Cell 14: Grand Prix Summary Dashboard
* **Tujuan:** Figur sinopsis seluruh Model Grand Prix.
* **Prosedur:**
  1. **Figur `fig_model_summary`** (2×2, 14×10):
     * Panel A: Radar chart (6 model × 5 metrik: Macro F1, per-class F1, 1−ECE).
     * Panel B: Grouped bar F1 per kelas per model, descending.
     * Panel C: One-vs-Rest ROC curves ConvNeXt vs LightGBM (per-class AUROC).
     * Panel D: McNemar heatmap $p$-value antar-pasangan model.

### Cell 15: Dynamic Gating Synthesis - Batas Atas Paradigma Konvensional
* **Tujuan:** Penyatuan GBDT + ConvNeXt via SNR-gating. Membuktikan ceiling dan justifikasi transisi ke Foundation Models.
* **Prosedur:**
  1. Skor kualitas: komplemen spectral flatness + rasio energi residu.
  2. Gating: $w_{\text{tab}}(s_i) = w_{\max} \cdot [1-\text{rank\_norm}(s_i)]^\gamma$.
  3. Two-Way Cendrawasih Shield.
* **Hasil:** OOF $0.9380$ - ceiling konvensional. External LB $0.59462$ membuktikan kerentanan domain shift. **→ Transisi ke Fase 5.**


## FASE 5: Bioacoustic Foundation Models & Grounding Taksonomi

### Cell 16: Google Perch v2 Extraction & Embedding Visualization *(dari RAM cache)*
* **Tujuan:** Ekstrasi embedding 1536-d + logit 14.795 taksa.
* **Prosedur:**
  1. TFHub `perch_v2/2`, inferensi Stream A (32 kHz, 3 window).
  2. Embedding L2-norm $\hat{e} \in \mathbb{S}^{1535}$ + norma pra-normalisasi + logit taksonomi.
  3. Peta indeks target: Bondol `[7329]`, Kutilang `[11701]`, 8 kandidat Paradisaeidae.
  4. **Figur `fig_model_perch_embedding`** (1×3, 16×5):
     * Panel A: t-SNE Perch 1536-d, diwarnai per spesies.
     * Panel B: t-SNE yang sama, diwarnai per source_group (bukti invarian mikrofon).
     * Panel C: t-SNE train + test overlay.
* **Hasil:** Zero-shot logit separability: Kutilang $+7.81$, Cendrawasih $+3.39$, Bondol $+0.36$.

### Cell 17: Multi-Taxa Paradisaeidae & RESOLVE Global Taxa Lock
* **Tujuan:** Diagnosa sub-spesies Cendrawasih + penguncian taksa global.
* **Prosedur:**
  1. AUROC zero-shot 8 taksa Paradisaeidae × 4 fold.
  2. Multi-Taxa Max-Pooling: $z_{\text{Cendra,pooled}}(x) = \max_{s \in \mathcal{S}_{\text{Para}}} z_s(x)$.
  3. RESOLVE: kunci ke *Cicinnurus regius* (idx 2893) setelah verifikasi AUROC.
  4. Inline: heatmap 8 taksa × 4 fold AUROC.
* **Hasil:** Pooled AUROC $0.8917$. Variansi taksa antar-fold dieliminasi.

### Cell 18: BirdNET v2.4 (48 kHz) Extraction *(dari RAM cache)*
* **Tujuan:** Juri forensik ortogonal independen (6.522 spesies, 48 kHz native).
* **Prosedur:**
  1. Inferensi Stream B (48 kHz, 3 window @ 3.0 s).
  2. Ekstrak 4 logit: $b_{\text{Bondol}}$, $b_{\text{Kutilang}}$, $b_{\text{Cendra}} = \max(\text{17 Paradisaeidae})$, $b_{\text{NonTarget,max}}$.
* **Output:** `birdnet_train_summary` (1826×4), `birdnet_test_summary` (814×4).

### Cell 19: Hybrid 1543-d Assembly, CosineHead Training & Calibration *(Full Retrain)*
* **Tujuan:** Rakit input hibrida, latih kepala klasifikasi, dan audit kalibrasi.
* **Prosedur:**
  1. Hybrid 1543-d = `[1536-d L2 emb]` ⊕ `[7-d Z-scored summary]`:
     * $z_{\text{Bondol}}, z_{\text{Kutilang}}, z_{\text{Cendra,pooled}}, z_{\text{NonTarget,max}}, \text{Top5 mean}, \Delta z_{\text{OOD}}, E_{\text{free}}$.
  2. **A/B Test:** Latih `PerchBottleneckCosineHead` 4-fold pada Pure 1536-d **dan** Hybrid 1543-d.
     * Arsitektur: `Linear(d→256) → LayerNorm → SiLU → Dropout(0.2) → CosineLinear(256→3, scale=16)`.
     * 40 epoch, CB-Focal Loss.
  3. Audit ECE (15 bins) + Brier Score.
  4. **Figur `fig_model_hybrid_calibration`** (2×2, 14×8):
     * Panel A: A/B bar chart (Pure vs Hybrid OOF F1, per-class breakdown).
     * Panel B: Learning curves overlay (val F1 per epoch, Pure vs Hybrid).
     * Panel C: Reliability diagram (confidence bins vs accuracy) Hybrid.
     * Panel D: Confidence histogram diwarnai benar/salah.
* **Hasil:** Pure $0.9239$, Hybrid $0.9314$ (+0.0075). ECE $0.0241$.


## FASE 6: Forensic Shielding & Error Analysis

### Cell 20: BirdNET In-Fold Threshold Calibration
* **Tujuan:** Kalibrasi ambang batas forensik secara in-fold bebas bocoran.
* **Prosedur:**
  1. Per fold $k$, hitung pada In-Fold Training Split:
     * $\tau_{\text{cendra}}^{(k)}$: P20 (retensi 80%) distribusi $b_{\text{Cendra}}$ kelas Cendrawasih.
     * $\tau_{\text{kutilang}}^{(k)}$: P5 (retensi 95%) distribusi $b_{\text{Kutilang}}$ kelas Kutilang.
  2. **Figur `fig_eval_birdnet_thresholds`** (1×3, 14×4):
     * Panel A: Violin $b_{\text{Cendra}}$ (True Cendra vs Non-Cendra), garis $\tau = -7.70$.
     * Panel B: Violin $b_{\text{Kutilang}}$, garis $\tau = -8.11$.
     * Panel C: Line chart stabilitas $\tau$ per fold.
* **Hasil:** $\bar{\tau}_{\text{cendra}} = -7.70 \pm 0.04$, $\bar{\tau}_{\text{kutilang}} = -8.11 \pm 0.04$. → `nested_forensic_thresholds.json`.

### Cell 21: Dual Forensic Shield Execution & Impact Analysis
* **Tujuan:** Eliminasi FP Cendrawasih + isolasi Trucuk.
* **Prosedur:**
  1. **Cendrawasih Veto:** Jika $\arg\max P = \text{Cendra}$ tapi $b_{\text{Cendra}} < \tau_{\text{cendra}}^{(k)}$ → reassign ke kelas ranking ke-2.
  2. **Trucuk Shield:** Tandai `is_trucuk = True` jika Kutilang confidence < 0.60 dan $b_{\text{Kutilang}} < \tau_{\text{kutilang}}^{(k)}$.
  3. **Figur `fig_eval_shield_impact`** (1×3, 16×5):
     * Panel A: Side-by-side confusion matrix (sebelum vs sesudah shield).
     * Panel B: Sankey flow diagram reassignment.
     * Panel C: 3 mel-spectrogram specimen error yang di-reassign.
* **Hasil:** 12/15 FP Cendrawasih dieliminasi. Presisi OOF Cendrawasih $77.8\% \to 89.6\%$. OOF F1 $0.9314 \to 0.9504$.


## FASE 7: Adaptasi Domain Transduktif

### Cell 22: 4-Gate Consensus Inlier Curation
* **Tujuan:** Menyaring pseudo-inliers berkemurnian $\approx 100\%$ dari data uji.
* **Prosedur:**
  1. kNN-5 cosine distance in-fold pada embedding mean-centered.
  2. **4-Gate Sterile Filter:**
     * Gate 1 (Confidence): $\max_c P_{\text{corr}} \ge Q_{0.55}(P_{\text{OOF}})$.
     * Gate 2 (Geometric): $d_{\text{perch}} \le Q_{0.60}(d_{\text{OOF}}) \land E_{\text{free}} \le Q_{0.85}(E_{\text{free,OOF}})$.
     * Gate 3 (Bilateral Cendra): $b_{\text{Cendra}} \ge -7.70$ jika label = Cendrawasih.
     * Gate 4 (Confounder): `is_trucuk == False`.
  3. **Figur `fig_eval_pseudo_quality`** (1×3, 16×4):
     * Panel A: Scatter confidence × kNN distance, hijau = inlier, merah = rejected.
     * Panel B: Funnel chart (814 → post-G1 → post-G2 → post-G3 → post-G4).
     * Panel C: Bar distribusi kelas inliers.
* **Hasil:** Kemurnian OOF $99.72\%$ (351/352). Round 1: 203 inliers.

### Cell 23: Transductive Self-Training Round 1 + Round 2 *(Full Retrain)*
* **Tujuan:** Domain adaptation via pseudo-inlier injection, 2 putaran.
* **Prosedur:**
  1. **Round 1:** Injeksi 203 inliers ke train split per fold ($\lambda=1.0$). Retrain CosineHead 40 epoch. Val split murni.
  2. Re-curation menggunakan Round 1 probs → 274 inliers.
  3. **Round 2:** Injeksi 274 inliers ($\lambda=0.85$). Retrain 40 epoch.
  4. Terapkan Forensic Shield ulang pada Round 2 output.
  5. Rekam: OOF F1 per round, FP Cendrawasih count, ECE.
* **Hasil:** R1 OOF F1 = $0.9521$, R2 OOF F1 = $\mathbf{0.9542}$. Bondol akurasi $100\%$. ECE $0.0219$.

### Cell 24: Round 3 Saturation Proof & Convergence Visualization
* **Tujuan:** Membuktikan stopping criterion secara matematis.
* **Prosedur:**
  1. Kurasi Round 3 → 292 inliers (+5 Cendrawasih marginal). Retrain $\lambda=0.80$.
  2. Prototype drift: $\Delta\mu_c = 1 - \cos(\mu_c^{(t)}, \mu_c^{(t-1)})$.
  3. **Figur `fig_eval_convergence`** (2×2, 14×10):
     * Panel A: OOF Macro F1 per round (R0→R1→R2→R3). R2 = puncak.
     * Panel B: Prototype drift bar per kelas per transisi.
     * Panel C: t-SNE R0 vs R2 vs R3.
     * Panel D: Mel-spectrogram 3 spesimen Cicadidae yang menyusup di R3.
* **Keputusan:** Round 2 = Pareto-optimal. R3 turun ke $0.9488$ (-0.0054). **Bobot R2 dikunci.**


## FASE 8: Architecture Optimization (Last Experiments)

> Setiap eksperimen memiliki *decision gate*: hanya konfigurasi yang melampaui baseline + 0.002 yang diadopsi. Seluruh eksperimen dari **full retrain** (tanpa load checkpoint).

### Cell 25: Exp-A - CosineHead Architecture Ablation
* **Tujuan:** Grid search konfigurasi head yang belum pernah diuji.
* **Prosedur:**
  1. Grid (Round 0 supervised only, 4-fold, tanpa self-training):
     ```
     hidden_dim: [128, 256, 384]
     dropout:    [0.1, 0.2, 0.3]
     scale:      [8, 16, 24]
     ```
     → 27 configs × 4 folds = 108 runs (~30 menit total, 1 run ≈ 15 detik).
  2. Ranking berdasarkan Global OOF Macro F1.
  3. **Gate A:** Jika top-1 > baseline (256, 0.2, 16) + 0.002 → adopt `BEST_HEAD`.
  4. Inline: heatmap ablation (dim × dropout, sliced per scale).
* **Output:** `BEST_HEAD_CONFIG` atau retain default.

### Cell 26: Exp-B - Embedding Augmentation + Exp-C - λ Ablation
* **Tujuan:** Dua eksperimen ringan dalam satu sel.
* **Prosedur - Exp-B:**
  1. Modifikasi training loop:
     ```python
     if training:
         # Augmentasi 1: Gaussian noise + Feature dropout
         x = x + torch.randn_like(x) * sigma
         mask = torch.bernoulli(torch.ones_like(x) * (1 - feat_drop))
         x = x * mask / (1 - feat_drop)
         # Augmentasi 2: Intra-class Mixup (khusus Cendrawasih, 83 sampel)
         if use_mixup:
             idx = torch.randperm(x.size(0))
             lam = torch.distributions.Beta(alpha, alpha).sample()
             x = lam * x + (1 - lam) * x[idx]
             # Label tetap (intra-class only: pasangan di-filter per kelas)
     ```
  2. Sweep tahap 1 - Noise+Dropout: $\sigma \in \{0.01, 0.02, 0.05\}$ × $\text{feat\_drop} \in \{0.0, 0.05, 0.10\}$ → 9 configs (Round 0, 4-fold).
  3. Sweep tahap 2 - Mixup (pada top-1 noise config): $\alpha \in \{0.2, 0.4, 0.6\}$ × scope $\in \{\text{all-class}, \text{Cendrawasih-only}\}$ → 6 configs.
     * *Justifikasi:* Gaussian noise menambah isotropic jitter, tapi Mixup menginterpolasi antar-sampel nyata di manifold - khususnya powerful untuk Cendrawasih yang hanya punya 83 sampel dari 4 source group. Keduanya komplementer.
  4. **Gate B:** Top-1 overall > baseline + 0.002 → adopt `BEST_AUG` (bisa noise-only, mixup-only, atau kombinasi).
* **Prosedur - Exp-C:**
  1. Gunakan `BEST_HEAD` + `BEST_AUG`. Jalankan full R0→R1→R2 pipeline.
  2. Sweep $\lambda_{\text{R2}} \in \{0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00\}$ → 7 configs.
  3. **Gate C:** $\lambda^* > 0.85$ + 0.002 → adopt `BEST_LAMBDA`.
* **Inline figur:** 2-panel - bar chart augmentation results + line chart λ vs F1.

### Cell 27: Exp-D - Inlier Ensemble (Perch × ConvNeXt)
* **Tujuan:** Blend probabilitas 3-kelas dari 2 arsitektur ortogonal. OOD tetap dari Perch.
* **Prosedur:**
  1. ConvNeXt OOF probs sudah tersedia dari Cell 12 (retrained). Jika tidak, retrain inference pass saja.
  2. Blend: `blended = w * perch_probs + (1-w) * conv_probs`.
  3. Sweep $w \in \{0.50, 0.55, ..., 0.95, 1.00\}$ → 11 configs.
  4. Terapkan Forensic Shield pada blended probs → evaluasi OOF F1.
  5. Analisis error overlap: Venn diagram error eksklusif Perch vs ConvNeXt.
  6. **Gate D:** Best $w < 1.0$ yang > baseline + 0.002 → adopt `BEST_W`. Jika semua $w < 1.0$ kalah → Perch solo champion.
  7. Jika Gate D passed, jalankan full R0→R2 pipeline dengan blended probs untuk curation.
* **Inline figur:** line chart $w$ vs F1 + Venn error.

### Cell 28: Exp-E - Unknown Rate Threshold Sweep
* **Tujuan:** Optimasi kuantil retensi OOF untuk batas Open-Set.
* **Prosedur:**
  1. Hitung skor anomali geometris $A_{\text{geo}}(x) = \tilde{d}_{\text{perch}}(x) + \alpha \cdot \max(0, E_{\text{free}}(x) - Q_{0.90}(E_{\text{free,OOF}}))$.
  2. **Sweep Geometric:** $p \in \{88, 89, ..., 98\}$ → 11 titik, hitung $\tau_p$ dan proporsi Unknown.
  3. **Sweep Taxonomic:** $p_{\text{Perch}} \times p_{\text{BirdNET}} \in \{3,5,7,10\}^2$ → 16 configs.
  4. Proxy metric: 4-class OOF Macro F1 (Unknown = sampel di-reject).
  5. **Figur `fig_exp_threshold_sweep`** (1×3, 16×4):
     * Panel A: Line chart Unknown-rate vs proxy-F1 (geometric).
     * Panel B: Heatmap $p_{\text{Perch}} \times p_{\text{BirdNET}}$ → proxy-F1 (taxonomic).
     * Panel C: Overlay posisi optimal kedua boundary.
  6. **Gate E:** Lock `BEST_GEO_P` dan `BEST_TAX_CONFIG`.


## FASE 9: Grand Synthesis & Submission

### Cell 29: Optimal Configuration Assembly & Final Pipeline
* **Tujuan:** Gabungkan semua keputusan Gate A–E + window pooling ablation, jalankan pipeline final dari awal.
* **Prosedur:**
  1. **Window Pooling Ablation** (5 menit, zero extra training):
     * Saat ini 3 VAD windows di-*mean-pool* → 1 embedding 1536-d. Uji 3 strategi:
       * **Mean pooling** (baseline): $\bar{e} = \frac{1}{3}\sum_{w=1}^{3} e_w$
       * **Max pooling**: $\bar{e}_j = \max_w e_{w,j}$ per dimensi
       * **Attention pooling**: $\bar{e} = \sum_w \alpha_w e_w$, di mana $\alpha_w = \text{softmax}(\|e_w\|_2)$ (higher-norm window = lebih vokal)
     * Evaluasi ketiga strategi pada Round 0 OOF (4-fold, `BEST_HEAD` config). Adopt jika Δ > +0.002.
     * *Justifikasi:* Mean pooling bisa dilute sinyal vokal jika 1 dari 3 window menangkap noise/silence. Attention pooling memberi bobot lebih ke window dengan energi vokal tertinggi.
  2. Cetak ringkasan konfigurasi final:
     ```
     FINAL CONFIG:
       Head:      BEST_HEAD_CONFIG  (Gate A)
       Augment:   BEST_AUG_CONFIG   (Gate B)
       Lambda R2: BEST_LAMBDA       (Gate C)
       Ensemble:  BEST_W_PERCH      (Gate D, atau 1.0)
       Geo p:     BEST_GEO_P        (Gate E)
       Tax cfg:   BEST_TAX_CONFIG   (Gate E)
       Pooling:   BEST_POOL         (mean/max/attention)
     ```
  3. Full pipeline dari Cell 19 output: R0 → Shield → R1 → Re-curation → R2 → Shield → OOD.
  4. Evaluasi Global OOF Macro F1 pasca-shield. **Target: > 0.9542.**

### Cell 30: Submission Export & Cross-Milestone Comparison
* **Tujuan:** Generate 5 submission CSV dari milestone pipeline, audit integritas, strategi submit.
* **Prosedur:**
  1. **Generate 5 submissions** dari checkpoint pipeline yang sudah terkomputasi:

     | # | File | Teknik | Sumber |
     |:---:|:---|:---|:---:|
     | 1 | `sub_perch_hybrid_baseline.csv` | Perch Hybrid 1543-d CosineHead + p95 OOD (tanpa Shield, tanpa ST) | Cell 19 |
     | 2 | `sub_forensic_shielded.csv` | + Dual Forensic Shield (Cendrawasih Veto + Trucuk) | Cell 21 |
     | 3 | `sub_selftrain_r2_shielded.csv` | + Self-Training R2 (274 inliers, λ=0.85) + Shield | Cell 23 |
     | 4 | `sub_optimized_geometric.csv` | Optimal Config A–E + **Geometric OOD** (`BEST_GEO_P`) | Cell 29 |
     | 5 | `sub_optimized_taxonomic.csv` | Optimal Config A–E + **Taxonomic OOD** (`BEST_TAX_CONFIG`) | Cell 29 |

  2. Setiap CSV: 814 baris, kolom `file_id,label`, label ∈ {Bondol Jawa, Kutilang, Cendrawasih, Unknown}. 0 NaN.
  3. **Tabel perbandingan** (printed + figur):
     * Per-submission: jumlah per kelas, Unknown count, Unknown rate (%).
     * Cohen's Kappa pairwise antara sub 3–5.
     * Row-by-row diff sub 3 vs sub 4 vs sub 5.
  4. **Figur `fig_final_predictions`** (2×3, 16×8):
     * Panel A: Donut distribusi prediksi sub 4 (geometric).
     * Panel B: Donut distribusi prediksi sub 5 (taxonomic).
     * Panel C: t-SNE Perch 1536-d - train + test (per prediksi sub 4) + Unknown (×).
     * Panel D: Stacked bar perbandingan 5 submissions (jumlah per kelas).
     * Panel E: Diff heatmap sub 4 vs sub 5 (agreement/disagreement per kelas).
     * Panel F: Progression line chart OOF F1 across 5 milestones.
  5. **Strategi submit ke Kaggle (max 5/hari):**
     * Prioritas 1: `sub_optimized_geometric.csv` (expected strongest)
     * Prioritas 2: `sub_optimized_taxonomic.csv` (diversifikasi OOD boundary)
     * Prioritas 3: `sub_selftrain_r2_shielded.csv` (safety fallback - proven 0.71221)
     * Sub 1–2 disimpan sebagai diagnostik, tidak perlu di-submit kecuali kuota sisa.
* **Artefak:** 5 CSV files + `submission_comparison_report.txt`.


## FASE 10: Delivery & Reproduksibilitas

### Cell 31: Master Ablation Table, Archiving & Hash Manifest
* **Tujuan:** Tabel sinopsis lengkap + bundel reproduksibilitas.
* **Prosedur:**
  1. **Figur `fig_master_ablation`** (single, 18×10):

| Tahap | Konfigurasi | OOF F1 | Cendra F1 (FP) | ECE | Status |
|:---|:---|:---:|:---:|:---:|:---|
| GBDT (Cell 11) | LightGBM 818-d | 0.8542 | 0.7410 (28) | 0.0512 | Baseline Tabular |
| AST (Cell 13) | BEATs 16 kHz | 0.8905 | 0.7410 (19) | 0.0440 | Dieliminasi |
| CNN (Cell 12) | ConvNeXt LP-FT 32 kHz | 0.9323 | 0.9448 (3) | 0.0315 | Best Vision |
| Gating (Cell 15) | ConvNeXt + GBDT | 0.9380 | 0.9480 (3) | 0.0289 | Ceiling Konvensional |
| Perch Pure (Cell 19) | 1536-d CosineHead | 0.9239 | 0.8312 (22) | 0.0264 | Ablasi |
| Perch Hybrid (Cell 19) | 1543-d CosineHead | 0.9314 | 0.8571 (15) | 0.0241 | Foundation Anchor |
| Shield (Cell 21) | + Dual Forensic | 0.9504 | 0.9189 (3) | 0.0228 | Shield Boost |
| R1 (Cell 23) | Self-Train 203 inliers | 0.9521 | 0.9295 (3) | 0.0223 | Domain Adapt |
| **R2 (Cell 23)** | **Self-Train 274 inliers** | **0.9542** | **0.9367 (2)** | **0.0219** | **Baseline Optimal** |
| R3 (Cell 24) | 292 inliers - Saturated | 0.9488 | 0.9012 (7) | 0.0275 | Bukti Saturasi |
| **FINAL (Cell 29)** | **Optimal Config A–E** | **?** | **?** | **?** | **Last Experiment** |

  2. SHA-256/MD5 manifest untuk semua artefak kunci.
  3. Zip ke `schematics_master_pipeline_artifacts.zip`.
  4. Cetak ringkasan keputusan Gate A–E (adopted/rejected + alasan).


## Estimasi Waktu Eksekusi (T4 GPU)

| Fase | Sel | Estimasi |
|:---|:---:|:---:|
| Setup + EDA | 1–7 | ~20 menit (I/O + PSD) |
| Preprocessing + FE + SHAP | 8–10 | ~15 menit |
| GBDT Grand Prix | 11 | ~7 menit |
| CNN Grand Prix (3 model × 4 fold × 23 ep) | 12 | ~120 menit |
| AST Investigation | 13 | ~40 menit |
| Summary + Gating | 14–15 | ~10 menit |
| Perch + BirdNET extraction | 16–18 | ~30 menit |
| Hybrid CosineHead A/B | 19 | ~10 menit |
| Forensic Shield | 20–21 | ~5 menit |
| Self-Training R1+R2+R3 | 22–24 | ~25 menit |
| **Exp-A: Head ablation (27 configs)** | **25** | **~30 menit** |
| **Exp-B+C: Aug + Mixup + λ (15+7 configs)** | **26** | **~60 menit** |
| **Exp-D: Ensemble blend** | **27** | **~15 menit** |
| **Exp-E: Threshold sweep** | **28** | **~5 menit** |
| Grand Synthesis (+ pooling ablation) + Export | 29–30 | ~20 menit |
| Delivery | 31 | ~3 menit |
| **Total** | | **~6 jam** |


## Pemetaan ~12 Figur Terbaik ke Laporan 10 Halaman

| Bab (Bobot) | Figur Pilihan | Hal. |
|:---|:---|:---:|
| **EDA (15%)** | `fig_eda_class_distribution`, `fig_eda_waveform_spectrogram`, `fig_eda_spectral_energy` | 1.5 |
| **Preprocessing (20%)** | `fig_preprocess_before_after`, `fig_feature_shap` | 1.5 |
| **Modeling (25%)** | `fig_model_cnn_learning`, `fig_model_summary`, `fig_model_perch_embedding` | 2.0 |
| **Evaluasi (30%)** | `fig_eval_folds`, `fig_eval_shield_impact`, `fig_eval_convergence`, `fig_exp_threshold_sweep` | 2.5 |
| **Struktur (10%)** | `fig_master_ablation` (tabel) | 0.5 |
| **Total** | **~12 figur** | **~8 hal. konten** |

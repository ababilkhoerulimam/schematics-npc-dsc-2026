# Master Blueprint: Day 4 Multi-Family Modular Pipeline (Tabular Triad + ConvNeXt-Tiny + Perch v2 + Transductive Adaptation)

**Proyek:** Schematics NPC DSC 2026 (Kompetisi Sains Data Nasional ITS)
**Tugas:** Audio Open-Set Classification (Bondol Jawa, Cendrawasih, Kutilang, dan Unknown)
**Target:** Menembus batas Public Leaderboard dari 0.66760 (sub_013) menuju 0.7200 - 0.7600 & Mengunci Laporan Teknis Publikasi (Bobot 65%)
**Dokumen:** `planning_day_4.md`
**Versi:** 2.0 (End-to-End Reproducible Master: 3-Pillar Multi-Family Benchmark, Deep EDA, In-Fold Taxa Audit, & Self-Training)
**Status:** Approved & Locked for Execution

---

## 1. Landasan Filosofis & Justifikasi Akademis (Defensible Against Committee Audit)

Blueprint Day 4 dirancang berjalan 100% mandiri (*self-contained*) di Kaggle GPU Tesla T4 dari Cell 1 hingga Cell 36 tanpa ketergantungan unggahan file eksternal manual.

### Menjawab Pertanyaan Kritis Juri ITS: "Mengapa Perch? Mengapa Bukan Tabular atau CNN Saja?"
Kami tidak membuang model konvensional untuk langsung melompat ke *black-box foundation model*. Seluruh arsitektur dibangun di atas **Multi-Family Empirical Grand Prix**:
1. **Pilar 1: Tabular Triad (818 Fitur Fisik Sinyal):** Membuktikan batas representasi 1D statistik (Centroid, Rolloff, ZCR, Autokorelasi). Menghasilkan OOF Macro F1 0.8463, namun rentan terhadap derau mikrofon recorder ($\eta^2 = 0.2262$).
2. **Pilar 2: Spatial 2D CNN (ConvNeXt-Tiny v2.0):** Membuktikan batas representasi visual spektrogram 2D (ImageNet pre-trained). Menghasilkan OOF Macro F1 0.9323, sangat presisi dalam mengidentifikasi pola spasial kicauan, namun tidak memiliki pemahaman semantik satwa liar.
3. **Pilar 3: Bioacoustic Foundation (Google Perch v2):** Membuktikan kekuatan representasi 14.795 taksa bioakustik dunia. Menghasilkan OOF Macro F1 0.9283 dan mendongkrak Public LB langsung ke 0.65+.
4. **Pilar 4: Transductive Inlierness Self-Training (Day 4 SOTA):** Menjembatani pergeseran domain akustik (*microphone domain shift*) antara data latih dan data uji tanpa kontaminasi label (*zero label poisoning*), menembus plafon kompetisi.

---

## 2. Master Configuration Flags (Modular Pipeline)

Semua pilar dikontrol melalui *modular flags* di Cell 1 sehingga dapat dieksekusi secara terpisah atau digabungkan secara penuh (*end-to-end*):

```python
# Master Pipeline Execution Switches
RUN_TABULAR_PILLAR   = True   # Ekstraksi 818 fitur & latih LightGBM + XGBoost + CatBoost (Ablasi)
RUN_CONVNEXT_PILLAR  = True   # Latih ConvNeXt-Tiny v2.0 LP-FT 2-Phase (Pilar Spasial)
RUN_PERCH_PILLAR     = True   # Latih Google Perch v2 Bottleneck Cosine Head (Pilar Bioakustik)
RUN_BIRDNET_FORENSIC = True   # Ekstraksi BirdNET v2.4 untuk Forensic Shielding
RUN_SELF_TRAINING    = True   # Inlierness-Guided Adaptive Pseudo-Labeling (Day 4 Innovation)
```

---

## 3. Struktur 7 Modul Pipeline (Total 36 Sel)

```
[Modul 1: Environment, Seeds, & Hardware Config] (Sel 1 - 2)
  └── Seed deterministik 42, CUBLAS workspace config, autodeteksi base directory.

[Modul 2: Deep Exploratory Data Analysis & Confounder Audit] (Sel 3 - 8)
  ├── Metadata Schema Audit, Class Imbalance Ratio (IR=13.54), Shannon Entropy.
  ├── Audio Streaminfo Header Audit, KS-Test Durasi, Deduplikasi Hash SHA256 (100% Unik).
  ├── Confounder Audit: Uji Kruskal-Wallis Eta-Squared (eta^2 = 0.2262 pada source_group).
  ├── Exact Combinatorial 4-Fold Group Partitioning (Cendrawasih: [35, 29, 16, 3]).
  ├── Deep Spectral Audit: Welch PSD Native 48k vs 32k (Pita 16-24 kHz <1.22% daya).
  └── Stereo Downmix Lossless Verification (Korelasi inter-channel 0.7702, selisih RMS 0.002).

[Modul 3: Pilar 1 - Tabular Triad Benchmark & Ablasi] (Sel 9 - 14)
  ├── Ekstraksi 818 Fitur DSP Preprocessing v2.0 (Spectral, Temporal, Cepstral, Rhythmic).
  ├── Pelatihan 4-Fold LightGBM (Leaf-wise GBDT, OOF F1 = 0.8431).
  ├── Pelatihan 4-Fold XGBoost (Depth-wise GPU, OOF F1 = 0.8399).
  ├── Pelatihan 4-Fold CatBoost (Symmetric Oblivious GPU, OOF F1 = 0.8292).
  ├── Ablasi Tabular & Dual-Tree Selection (50% LGBM + 50% XGBoost, F1 = 0.8463, CatBoost dipangkas).
  └── Ekspor Artefak Tabular OOF dan Test (`oof_p_tabular_v2.npy`, `test_p_tabular_v2.npy`).

[Modul 4: Pilar 2 - Spatial 2D CNN ConvNeXt-Tiny v2.0] (Sel 15 - 17)
  ├── GPU Log-Mel Frontend (128 bin), Spectral Subtraction p10, SpecAugment, Focal Loss.
  ├── Pelatihan 4-Fold ConvNeXt-Tiny v2.0 (LP-FT 2-Phase, OOF F1 = 0.9323, Cendra TP=77, FP=3).
  └── 3-Crop TTA Multi-Window Inference, Temperature Sharpening (T=0.55), In-Fold kNN-5 Distance.

[Modul 5: Pilar 3 - Bioacoustic Foundation Google Perch v2] (Sel 18 - 22)
  ├── Preprocessing v2.1: HighPass 100Hz, Linear Peak Norm 0.95, Dynamic Equidistant 3-Window 5s.
  ├── Load Model Perch v2 TFHub & Dynamic Taxa Resolution eBird labels.csv.
  ├── Batch Inference 1.826 Train & 814 Test (Embedding 1536-d & 16-d Logit Summary).
  ├── Out-of-Fold Taxa Screening (AUROC in-fold verification pada Cendrawasih).
  └── PerchBottleneckCosineHead Training (40 Epochs, Zero-Peeking Final Epoch, OOF F1 = 0.9283).

[Modul 6: Multi-Family Ensembling & Forensic Taxonomic Shield] (Sel 23 - 27)
  ├── SNR-Dynamic Gating Master Optimization (Evaluasi premise bising Perch vs Conv vs Tabular).
  ├── Load Official BirdNET v2.4 (LiteRT) & Batch Logit Extraction (Bondol, Kutilang, Cendra, NonTarget).
  ├── Forensic Reassignment Cendrawasih (thr=-7.70) ke second-best class (OOF F1 naik ke 0.9482).
  ├── Forensic Trucuk Shield (Pycnonotus goiavier thr=-8.11) ke Unknown.
  └── Grand Prix Comparison Matrix (Tabel evaluasi seluruh model untuk Bab 4 Laporan).

[Modul 7: Transductive Adaptive Self-Training & Multi-Submission Export] (Sel 28 - 36)
  ├── Helmholtz Free-Energy Scoring E(x) & Manifold Distance Evaluation.
  ├── Adaptive Test Inlierness Curation (Seleksi Top 15% - 20% test soundscapes paling murni).
  ├── Augmented Dataset Construction (1.826 Train + N_pseudo Test).
  ├── Retraining Bottleneck Cosine Head (Self-Training 40 Epochs).
  ├── Test Inference & Calibrated Post-Processing.
  ├── Sweet-Spot Manifold Unknown Locking (~43.5% Total Unknown).
  ├── Quality Assertion & Submission Export: sub_014, sub_015, sub_016.
  ├── Automated Publication Diagnostics Plot (ood_dual_manifold_diagnostics.png).
  └── Final Working Directory Compression & Download Link (schematics_day4_all_artifacts.zip).
```

---

## 4. Rincian Sel-demi-Sel & Expected Verified Outputs

### Modul 1: Environment, Seeds, & Hardware Config (Sel 1 - 2)

* **Cell 1: Environment Setup, Deterministic Seeds, and Hardware Config**
  * *Operasi:* Set seed 42 (PyTorch, CuDNN, NumPy). Inisialisasi CUDA device Tesla T4. Set master configuration flags.
  * *Expected Output:* `hardware_target | device=cuda:0 | gpu=Tesla T4 | free_vram=14.46GB`.

* **Cell 2: Remote Dataset Auto-Discovery & Path Normalization**
  * *Operasi:* Autodeteksi base directory `/kaggle/input/competitions/schematics-2026` atau fallback lokal.
  * *Expected Output:* `dataset_discovered | base_dir=/kaggle/input/... | work_dir=/kaggle/working`.

---

### Modul 2: Deep Exploratory Data Analysis & Confounder Audit (Sel 3 - 8)

* **Cell 3: Metadata Schema Audit & Cardinality Verification**
  * *Operasi:* Load `train/metadata.csv` (1.826) dan `sample_submission.csv` (814). Assert 0 null. Hitung IR dan Shannon entropy.
  * *Expected Output:* `train_shape=(1826, 4) | test_shape=(814, 2) | IR=13.54 | entropy=1.163 bits`.

* **Cell 4: Audio Header & Exact Deduplication Audit**
  * *Operasi:* Parse streaminfo FLAC 48 kHz stereo 24-bit. Audit exact SHA256 deduplication (100% unique). KS-test durasi.
  * *Expected Output:* `audio_integrity_verified | duplicates=0 | sr=48000 | ks_test_p=0.6695`.

* **Cell 5: Confounder Audit: Kruskal-Wallis Eta-Squared Test**
  * *Operasi:* Uji Kruskal-Wallis pada fitur akustik antar `source_group`. Hitung effect size $\eta^2$.
  * *Expected Output:* `confounder_audit | eta_squared=0.2262 | p_value=5.77e-53 (Statistically Significant Confounder)`.

* **Cell 6: Exact Balanced Group Partitioning (4-Fold StratifiedGroupKFold)**
  * *Operasi:* Optimasi kombinatorial eksak memetakan 15 `source_group` ke 4 fold tanpa kebocoran mikrofon.
  * *Expected Output:* `partition_locked | Cendrawasih=[35, 29, 16, 3] | group_leakage=0`.

* **Cell 7: Deep Spectral Audit: Welch PSD & Nyquist 48k vs 32k**
  * *Operasi:* Analisis spektrum daya Welch pada pita frekuensi: 0-2k, 2-8k, 8-16k, dan 16-24k.
  * *Expected Output:* `spectral_audit: band_16_24k < 1.22% power | band_8_16k up to 9.08% on Cendrawasih`.

* **Cell 8: Stereo Downmix Lossless Verification & EDA Plots Export**
  * *Operasi:* Audit selisih RMS stereo (0.002) dan korelasi channel (0.7702). Simpan visualisasi 3-panel EDA.
  * *Expected Output:* `eda_acoustic_profiles.png saved | stereo_lossless_verified=True`.

---

### Modul 3: Pilar 1 - Tabular Triad Benchmark & Ablasi (Sel 9 - 14)

* **Cell 9: Tabular Signal Extraction (818 Fitur DSP Preprocessing v2.0)**
  * *Operasi:* Ekstraksi 818 dimensi fitur: Spectral Centroid, Rolloff, Contrast, Mel statistical summaries, ZCR, RMS.
  * *Expected Output:* `tabular_extracted: X_train=(1826, 818) | X_test=(814, 818) | elapsed ~45s`.

* **Cell 10: 4-Fold LightGBM (Leaf-Wise GBDT)**
  * *Operasi:* Pelatihan 4-fold LightGBM dengan balanced class weights.
  * *Expected Output:* `lgbm_trained | OOF Macro F1 = 0.8431 | Cendra FP = 36`.

* **Cell 11: 4-Fold XGBoost (Depth-Wise GPU)**
  * *Operasi:* Pelatihan 4-fold XGBoost GPU histogram-based.
  * *Expected Output:* `xgboost_trained | OOF Macro F1 = 0.8399 | Cendra FP = 34`.

* **Cell 12: 4-Fold CatBoost (Symmetric Oblivious GPU)**
  * *Operasi:* Pelatihan 4-fold CatBoost GPU.
  * *Expected Output:* `catboost_trained | OOF Macro F1 = 0.8292 | Cendra FP = 51 (Ablation Defect Detected)`.

* **Cell 13: Tabular Ablasi & Dual-Selection Ensembling**
  * *Operasi:* Evaluasi perbandingan Grand Prix tabular. Pangkas CatBoost karena over-rejection minoritas. Kunci 50% LGBM + 50% XGBoost.
  * *Expected Output:* `dual_tabular_blend: OOF Macro F1 = 0.8463 (CatBoost pruned successfully)`.

* **Cell 14: Simpan Artefak Probabilitas Tabular**
  * *Operasi:* Simpan `oof_p_tabular_v2.npy`, `test_p_tabular_v2.npy`, dan `tabular_triad_summary.json`.
  * *Expected Output:* `artifacts_saved: oof_p_tabular_v2.npy, test_p_tabular_v2.npy`.

---

### Modul 4: Pilar 2 - Spatial 2D CNN ConvNeXt-Tiny v2.0 (Sel 15 - 17)

* **Cell 15: GPU Mel-Spectrogram Frontend & VAD Jittered Augmentation**
  * *Operasi:* Inisialisasi frontend 128 mel-bins, stationary spectral subtraction p10, dan SpecAugment 2D.
  * *Expected Output:* `frontend_ready: logmel_128_bins | spec_sub_p10=True | spec_augment=True`.

* **Cell 16: Live Retraining ConvNeXt-Tiny v2.0 (LP-FT 2-Phase)**
  * *Operasi:* Pelatihan 4-fold ConvNeXt-Tiny v2.0 (Phase 1: Linear Probe 5 ep, Phase 2: Fine-Tuning 20 ep).
  * *Expected Output:* `convnext_retrained: Global OOF Macro F1 = 0.9323 | Cendra F1 = 0.9448 (TP=77, FP=3)`.

* **Cell 17: 3-Crop TTA Multi-Window Inference & In-Fold Manifold Distance**
  * *Operasi:* Evaluasi 3-crop TTA pada 814 data uji dengan kalibrasi $T=0.55$. Hitung jarak kNN-5 in-fold mean-centered.
  * *Expected Output:* `convnext_tta_evaluated | oof_d_conv mean=0.0724 | test_d_conv mean=0.1635`.

---

### Modul 5: Pilar 3 - Bioacoustic Foundation Google Perch v2 (Sel 18 - 22)

* **Cell 18: DSP Preprocessing v2.1 (Linear Peak Norm & Dynamic 3-Window 5s)**
  * *Operasi:* HighPass 100 Hz, linear peak norm 0.95 (tanpa tanh), potong 3-window 5.0s (160.000 sampel @ 32 kHz).
  * *Expected Output:* `dsp_v2.1_cached | train=1826 | test=814 | zero_pad_corruption=0.0%`.

* **Cell 19: Load Google Perch v2 from TFHub & Dynamic Taxa Resolution**
  * *Operasi:* Muat model `perch_v2/2` dan cocokkan regex taksa eBird `labels.csv`.
  * *Expected Output:* `perch_loaded | emb_dim=1536 | taxa: Bondol=7329, Kutilang=11701, Cendra=(10 cands)`.

* **Cell 20: Batch Feature Extraction Perch v2 (Train & Test)**
  * *Operasi:* Ekstraksi batch 32 menghasilkan embedding (1826, 3, 1536) dan ringkasan logit kandidat 16-d.
  * *Expected Output:* `perch_features_extracted | train_emb=(1826, 3, 1536) | test_emb=(814, 3, 1536)`.

* **Cell 21: Out-of-Fold Taxa Screening & Verification**
  * *Operasi:* Seleksi taksa Cendrawasih murni in-fold via AUROC validasi (mengunci indeks resmi eBird 2893).
  * *Expected Output:* `verified_cendra_idx=2893 | in_fold_auroc=0.8911 | final_summary_locked=(1826, 3, 7)`.

* **Cell 22: PerchBottleneckCosineHead Training (Zero-Peeking Evaluation)**
  * *Operasi:* Pelatihan 40 epoch head Cosine dengan MIL Softmax Bag Pooling. Evaluasi wajib final epoch 40.
  * *Expected Output:* `perch_head_trained | Global OOF Macro F1 = 0.9283 | Bondol=0.9838, Kutilang=0.9968, Cendra=0.8043`.

---

### Modul 6: Multi-Family Ensembling & Forensic Taxonomic Shield (Sel 23 - 27)

* **Cell 23: SNR-Dynamic Gating Master Optimization**
  * *Operasi:* Evaluasi empiris premis bising Perch vs Conv vs Tabular. Grid search parameter gating optimal.
  * *Expected Output:* `snr_gating_locked | r_perch=0.70, r_conv=0.15, w_tab=0.15 | Master OOF F1 = 0.9414`.

* **Cell 24: Load Official BirdNET v2.4 (LiteRT) & Batch Logit Extraction**
  * *Operasi:* Inisialisasi pustaka `birdnet` v1.3.0. Ekstraksi logit 3-window 48 kHz (Bondol, Kutilang, Cendra, NonTarget).
  * *Expected Output:* `birdnet_extracted: train=(1826, 3, 4) | test=(814, 3, 4) | elapsed ~9m`.

* **Cell 25: Forensic Cendrawasih Fallback Reassignment (thr = -7.70)**
  * *Operasi:* Reassign panggilan Cendrawasih bising Perch ke kelas kedua jika BirdNET logit $< -7.70$.
  * *Expected Output:* `cendra_reassigned ~89 samples | OOF Macro F1 jumps to 0.9482 (+0.0199)`.

* **Cell 26: Forensic Trucuk / Alien Bulbul Shield (thr = -8.11)**
  * *Operasi:* Isolasi 72 file Kutilang bising Perch dengan detektor BirdNET mati ke kandidat Unknown.
  * *Expected Output:* `trucuk_shielded ~72 samples to Unknown | zero_contamination_to_bondol=True`.

* **Cell 27: Multi-Family Grand Prix Matrix Compilation**
  * *Operasi:* Rangkum performa OOF dan karakteristik seluruh model ke dalam satu tabel komparatif laporan.
  * *Expected Output:* `model_grand_prix_summary.json saved | 4 families benchmarked`.

---

### Modul 7: Transductive Adaptive Self-Training & Multi-Submission Export (Sel 28 - 36)

* **Cell 28: Helmholtz Free-Energy Scoring E(x) & Manifold Distance Evaluation**
  * *Operasi:* Hitung skor energi bebas $E(x) = -T \log \sum \exp(z_c / T)$ dan jarak kNN-5 mean-centered.
  * *Expected Output:* `energy_scores_computed | oof_energy_mean < test_energy_mean`.

* **Cell 29: Adaptive Test Inlierness Curation (Anti-Poisoning Filter)**
  * *Operasi:* Seleksi otomatis 120-150 sampel test dengan 3 filter konsensus (Top 25% prob + Top 40% distance + BirdNET verification).
  * *Expected Output:* `curated_pseudo_samples: N in [120, 160] | zero_unknown_overlap=True`.

* **Cell 30: Augmented Dataset Construction & Class Re-balancing**
  * *Operasi:* Gabungkan 1.826 train asli dengan $N_{\text{pseudo}}$ test terkurasi. Hitung inverse-frequency class weights.
  * *Expected Output:* `augmented_dataset_shape=(1826 + N_pseudo, 3, 1536)`.

* **Cell 31: Retraining Bottleneck Cosine Head (Self-Training 40 Epochs)**
  * *Operasi:* Latih ulang head klasifikasi selama 40 epoch pada augmented manifold menggunakan Cosine Annealing LR.
  * *Expected Output:* `self_training_completed | elapsed < 45s | final_loss < 0.05`.

* **Cell 32: Self-Trained Test Inference & Calibrated Probability Matrix**
  * *Operasi:* Forward pass pada 814 data uji menghasilkan probabilitas teradaptasi domain.
  * *Expected Output:* `p_selftrained_test shape=(814, 3)`.

* **Cell 33: Sweet-Spot Manifold Unknown Quantile Locking (~43.5% Target)**
  * *Operasi:* Kunci ambang batas $\tau$ sedemikian rupa sehingga total Unknown (termasuk Trucuk shield) berada di 43.0% - 44.0%.
  * *Expected Output:* `tau_locked | total_unknown_count ~350-357 (43.0% - 43.8%)`.

* **Cell 34: Multi-Submission Assembly & Verification (sub_014 & sub_015)**
  * *Operasi:* Ekspor berkas submission resmi `experiment 14.csv` dan `experiment 15.csv`. Verifikasi 814 baris, 0 null, 4 kelas.
  * *Expected Output:* `experiment 14.csv & experiment 15.csv exported successfully | assertions passed`.

* **Cell 35: Automated Publication Diagnostics Plot**
  * *Operasi:* Simpan grafik 2-panel publikasi (`ood_dual_manifold_diagnostics.png`) untuk Bab 3 & 5 Laporan.
  * *Expected Output:* `ood_dual_manifold_diagnostics.png saved`.

* **Cell 36: Full Working Directory Compression & Download Link**
  * *Operasi:* Kompres seluruh isi `/kaggle/working` menjadi `schematics_day4_all_artifacts.zip` dan tampilkan `FileLink`.
  * *Expected Output:* `schematics_day4_all_artifacts.zip created | download link displayed`.

---

## 5. Matriks Risiko & Protokol Mitigasi

| Potensi Risiko | Probabilitas | Dampak | Protokol Mitigasi Terverifikasi |
|---|---|---|---|
| **Label Poisoning pada Pseudo-Labeling** | Sedang | Fatal (Drop LB) | Menerapkan 3 lapis filter konsensus: Top 25% confidence + Top 40% manifold distance terdekat + verifikasi aktif logit BirdNET. |
| **Over-Rejection pada Cendrawasih** | Rendah | Sedang | Cendrawasih yang ditolak BirdNET tidak dibuang ke Unknown, melainkan dikembalikan ke kelas kedua (*second-best class*), menjaga F1 OOF di 0.9482. |
| **Shake-Down di Private LB (70%)** | Rendah | Fatal | Ambang batas Unknown dikalibrasi berbasis kuantil OOF In-Distribution dan kurva sweet-spot konsisten Day 1-3, bukan fitting acak ke 244 baris Public LB. |
| **Kaggle Kernel Timeout / Out of Memory** | Rendah | Rendah | Menggunakan TFLite LiteRT dan checkpointing batch (CHUNK_FILES=64) yang terbukti selesai dalam <10 menit di Day 3. |

---

## 6. Kepatuhan Penuh Terhadap Regulasi Kompetisi (Guidebook ITS)

1. **Strict Zero External Raw Audio:** Seluruh pelatihan fitur berasal dari 1.826 train audio resmi panitia dan 814 audio uji transduktif. Nol raw audio eksternal ditambahkan ke dataset.
2. **Atribusi Pretrained Publik Transparan (Guidebook Hal 10):**
   * *Google Perch v2:* Google Research & Cornell Lab of Ornithology (Kaggle Models/TFHub).
   * *BirdNET v2.4:* K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology (PyPI official package `birdnet` v1.3.0).
3. **Reprodusibilitas Penuh:** Kunci `seed=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, dan isolasi 4-fold tanpa kebocoran data.

Blueprint Day 4 v2.0 ini resmi disetujui, terkunci di `schematics_day4_all_artifacts/planning_day_4.md`, dan siap dieksekusi sel-demi-sel secara interaktif.

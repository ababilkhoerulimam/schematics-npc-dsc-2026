# Preprocessing & Methodology Plan

**Project:** Schematics NPC DSC 2026 - Bird Chirp Openset Classification
**Version:** 1.2 (revised: 32 kHz default SR, ImageNet 2D CNN backbones, 256-d bottleneck OOD embedding)
**Date:** 2026-10-15
**Maturity:** Proposal (belum dieksekusi, menunggu validasi empiris)

---

## Case Signature

Klasifikasi audio 4-kelas (3 known + 1 Unknown) dengan karakteristik:

- **Extreme class imbalance:** IR = 13.54 (Bondol Jawa 1124 vs Cendrawasih 83) [E-011]
- **Open-set rejection:** Kelas Unknown tanpa contoh berlabel di train [Kompetisi]
- **Group-structured data:** 15 source groups pure per-class, minimum 4 groups per kelas [E-008, E-010]
- **Field recording noise:** Clipping 40-55%, silence ~10%, noise lingkungan hutan tropis [E-015]
- **Distinctive spectral signatures:** Kutilang tonal 2-4 kHz, Bondol broadband 2-6 kHz, Cendrawasih bimodal 0-2 kHz + 6-8 kHz [E-016]
- **Redundant stereo:** Korelasi antar-channel 0.724-0.800, delta RMS ~0.002 [E-018]
- **Negligible ultrasonic content:** Band 16-24 kHz hanya 0.11-1.22% energi total, namun band 8-16 kHz membawa hingga 9.08% energi harmonik Cendrawasih [E-017]

---

## Fase 0: Partisi Data

**Metode:** 4-fold StratifiedGroupKFold pada `source_group`, stratified by `label`.

**Justifikasi:** Cendrawasih dan Kutilang masing-masing punya tepat 4 source groups [E-010]. 4-fold membagi tepat 1 group per fold per kelas minoritas dengan zero recording-source leakage [D-001].

**Output:** `metadata_with_folds.csv` dengan kolom `fold` (0-3).

**Validasi:**
- Setiap fold memuat representasi ketiga kelas.
- Zero group overlap antar train-val split.
- Distribusi label per fold proporsional.

**Peringatan Kardinalitas [FIX-A]:** Cendrawasih memiliki grup berukuran 3, 16, 29, 35 [E-014]. Fold yang mendapat grup N=3 hanya punya 3 sampel validasi Cendrawasih, menyebabkan 1 kesalahan prediksi menjatuhkan F1 Cendrawasih fold tersebut ~33.3%. Oleh karena itu, Macro F1 **WAJIB** dihitung secara Global OOF (kumpulkan seluruh 1826 prediksi OOF, hitung Macro F1 sekaligus), **BUKAN** rata-rata F1 per fold.

---

## Fase 1: Preprocessing Pipeline

### 1.1 Mono Downmix

```
output = (ch0 + ch1) / 2
```

| Parameter | Nilai | Justifikasi |
|---|---|---|
| Method | Arithmetic mean | Korelasi stereo 0.724-0.800 dan delta RMS ~0.002 menjadikan kedua channel redundan [E-018, D-006] |
| Penghematan | 50% ukuran tensor input | Tanpa degradasi sinyal kicauan |

### 1.2 Resampling

| Parameter | Nilai | Justifikasi |
|---|---|---|
| Target SR (Default - 2D CNN) | 32.000 Hz | Mempertahankan harmonik 8-16 kHz (hingga 9.08% energi Cendrawasih) [E-017, D-005]. Arsitektur 2D CNN (EfficientNet/ConvNeXt/ResNet) fleksibel terhadap panjang frame temporal |
| Target SR (Ablasi / Audio-specific) | 16.000 Hz | Digunakan khusus untuk eksperimen ablasi ABL-05 atau jika menguji EfficientAT (`mn10_as`) |
| Metode | `scipy.signal.resample_poly(up, down)` dengan polyphase anti-aliasing filter | [FIX-B] `librosa.resample` (kaiser_best) masih valid sebagai alternatif. `resample_poly` dipilih karena menghindari Gibbs phenomenon / ringing overshoot pada rekaman yang sudah clipping (peak > 1.0 terdeteksi di test set, E-015). Untuk 48k->32k: `resample_poly(x, up=2, down=3)`. Untuk 48k->16k: `resample_poly(x, up=1, down=3)` |

`target_sr` adalah parameter pipeline, bukan hardcoded. **Default diset ke 32 kHz** untuk memaksimalkan informasi frekuensi tinggi pada kelas minoritas Cendrawasih.

### 1.3 Peak Normalization

```
x_norm = x / max(|x|)
```

| Parameter | Nilai | Justifikasi |
|---|---|---|
| Method | Per-sample peak normalization ke [-1, 1] | 40-55% rekaman mengalami clipping digital >= 0.99 [E-015]. [FIX-C] Normalisasi peak menyeragamkan variasi gain mikrofon-burung (gain invariance) dan memotong noise floor ekstrem. **BUKAN** untuk menghilangkan distorsi non-linear clipping yang sudah terjadi di perangkat sensor. FFT adalah operasi linier: F{a*x(t)} = a*F{x(t)}, sehingga mengalikan sinyal ter-clip dengan skalar tidak menghapus harmonik distorsi clipping |
| Alternatif ditolak | Soft peak -1.0 dBFS | Membutuhkan konversi dB yang menambah kompleksitas tanpa keuntungan terukur |

### 1.4 Standardisasi Durasi (10.0 Detik)

| Kondisi | Strategi | Justifikasi |
|---|---|---|
| Audio < 10s | Reflect-pad (memantulkan sinyal) | Durasi minimum 5.27s [E-009]. Reflect-pad menghindari artefak periodesitas palsu yang muncul pada wrap-around repeat dan menghindari injeksi silence pada zero-pad |
| Audio > 10s (train) | Random crop 10s | Augmentasi implisit posisi temporal |
| Audio > 10s (test) | Center crop 10s | Deterministik dan reproducible |

Jumlah sampel pada 32 kHz (default): 320.000. Pada 16 kHz (ablasi): 160.000.

### 1.5 Log-Mel Spectrogram

| Parameter | Nilai (32 kHz - Default) | Nilai (16 kHz - Ablasi) | Justifikasi |
|---|---|---|---|
| n_fft | 2048 (~64 ms) | 1024 (~64 ms) | Resolusi frekuensi cukup untuk memisahkan band 2-4 kHz Kutilang dari 4-6 kHz Bondol [E-016] |
| hop_length | 512 (~16 ms) | 320 (~20 ms) | 75% overlap, resolusi temporal memadai untuk pola repetitif kicauan |
| n_mels | 128 | 128 | Standar untuk audio classification, resolusi mel cukup untuk 3 kelas dengan signature spektral berbeda |
| fmin | 200 Hz | 200 Hz | Memotong low-frequency rumble (angin, getaran tanah, hum sensor) yang tidak informatif |
| fmax | 16000 Hz | 8000 Hz | Disesuaikan Nyquist; pada 32 kHz mempertahankan seluruh band informatif termasuk 8-16 kHz [E-017] |
| Konversi dB | power_to_db, ref=max, top_db=80 | Idem | Kompresi range dinamis untuk representasi visual dan input neural network |

**Dimensi output:**
- **32 kHz (Default):** `128 x 626` (128 mel bins x ceil(320000/512) time frames).
- **16 kHz (Ablasi):** `128 x 501` (128 mel bins x ceil(160000/320) time frames).

### 1.6 Per-Sample Normalization

```
S_norm = (S_dB - mean(S_dB)) / (std(S_dB) + 1e-6)
```

Normalisasi z-score per spektrogram individual. Meredam variasi gain mikrofon lapangan tanpa memerlukan statistik global dataset (fold-safe by design). **Penting:** Jangan menerapkan normalisasi mean/std RGB ImageNet (`[0.485, 0.456, 0.406]`) setelah tahap ini agar tidak merusak skala Z-score per sampel.

### 1.7 Urutan Eksekusi Pipeline dalam `Dataset.__getitem__`

```
Raw Stereo FLAC
--> 1.1 Mono Downmix
--> 1.2 Resample (polyphase anti-aliased, default 32 kHz)
--> 1.3 Peak Normalize [-1, 1]
--> 1.4 Duration Standardize (reflect-pad / random crop)
--> [Train only] 2.2 Waveform Mixup
--> [Train only] 2.3 Random Gain Jitter
--> [Train only] 2.4 Additive Gaussian Noise
--> 1.5 Log-Mel Spectrogram
--> [Train only] 2.1 SpecAugment (on spectrogram)
--> 1.6 Per-Sample Z-Score Normalization
```

Urutan ini memastikan: (a) Gain Jitter mengubah SNR riil sebelum mel extraction sehingga efeknya tidak terhapus oleh Z-score, (b) Mixup terjadi di domain waveform (campuran akustik koheren), (c) SpecAugment diterapkan pada spectrogram setelah mel extraction.

---

## Fase 2: Augmentasi (Train Split Saja)

Semua augmentasi diterapkan secara stokastik per batch saat training. Tidak pernah diterapkan pada data validasi atau test.

### 2.1 SpecAugment

| Parameter | Nilai | Justifikasi |
|---|---|---|
| Frequency masking | 2 strip, maks lebar 16 bin | Memaksa invariansi terhadap spike bersama ~2.9 kHz yang muncul di semua kelas [E-016] |
| Time masking | 2 strip, maks lebar 32 frame | Mensimulasikan jeda alami antar kicauan burung |
| Sumber | Park et al. 2019 (SpecAugment) | Standar augmentasi audio classification |

### 2.2 Waveform Mixup

| Parameter | Nilai | Justifikasi |
|---|---|---|
| Distribusi | Beta(0.2, 0.2) | Alpha rendah menghasilkan campuran yang didominasi salah satu sampel, menghindari artefak akustik ambigu |
| Domain penerapan | Waveform (sebelum mel extraction) | Campuran di domain waktu lebih koheren secara akustik dibanding interpolasi langsung pada spektrogram |
| Target | Mitigasi IR = 13.54 [E-011] | Memperluas ruang representasi kelas minoritas (Cendrawasih 83 sampel) tanpa data eksternal |
| Ablasi wajib | Bandingkan Macro F1 OOF dengan dan tanpa Mixup | Jika tidak meningkatkan performa, buang |

### 2.3 Random Gain Jitter

| Parameter | Nilai | Justifikasi |
|---|---|---|
| Range | +/- 6 dB (faktor 0.5x - 2.0x) | Mensimulasikan variasi jarak mikrofon-burung di lapangan |
| Penerapan | Pada waveform sebelum mel extraction | Menjaga konsistensi dengan normalisasi peak |

### 2.4 Additive Gaussian Noise

| Parameter | Nilai | Justifikasi |
|---|---|---|
| SNR range | 15-30 dB | Mensimulasikan variasi noise lingkungan hutan tropis (hujan, serangga, angin) [E-015] |
| Penerapan | Pada waveform sebelum mel extraction | Noise aditif di domain waktu secara fisik memodelkan kondisi perekaman nyata |

### Augmentasi yang Sengaja Tidak Dipakai

| Teknik | Alasan Penolakan |
|---|---|
| Time stretching | Mengubah durasi dan rate temporal kicauan, merusak pola repetisi yang menjadi fitur pembeda |
| Pitch shifting | Menggeser frekuensi fundamental kicauan, merusak identitas spektral spesies [E-016] |
| Random EQ/filtering | Berisiko menghapus band frekuensi yang justru menjadi pembeda utama antar spesies |
| HorizontalFlip pada spectrogram | [FIX-E] Sumbu X = waktu: membalik urutan kicauan secara temporal tidak bermakna secara akustik dan menghasilkan pola yang tidak pernah terjadi di alam |
| VerticalFlip pada spectrogram | [FIX-E] Sumbu Y = frekuensi: membalik frekuensi rendah-tinggi menghancurkan seluruh signature spektral spesies |

---

## Fase 3: Modeling (2-Track)

### Track A: Classical ML Baseline (CPU)

**Tujuan:** Pembanding wajib untuk rubrik Modeling (25%) dan baseline Macro F1.

**Fitur:**
- Summary statistics per mel bin: mean, std, skewness, kurtosis
- Delta (first-order temporal difference) statistics
- Delta-delta (second-order) statistics
- Spectral descriptors: centroid, bandwidth, rolloff, flatness, ZCR
- Estimasi total: 500-800 fitur per audio

**Model:** LightGBM dengan `class_weight='balanced'` atau `scale_pos_weight` per kelas.

**Validasi:** 4-fold StratifiedGroupKFold OOF, metrik Global Macro F1.

### Track B: Deep Learning (GPU)

[D-007] Mengingat ukuran dataset yang kecil ($N=1.826$, Cendrawasih $N=83$ [E-007]), arsitektur Transformer murni (ViT/AST ~87M, BEATs ~90M) **dieliminasi** karena risiko *overfitting* parah dan *curse of dimensionality* pada ruang fitur OOD. Sebagai gantinya, digunakan arsitektur **ImageNet-pretrained 2D CNN** yang memiliki *inductive bias* spasial kuat dan efisiensi parameter tinggi:

| Prioritas | Model & Sumber Korpus | Parameter | Spesifikasi & Justifikasi |
|---|---|---|---|
| **Utama (B1)** | **EfficientNet-B0** (`cv_arch_efficientnet_compound_scaling`) | ~5.3M | Hanya gunakan varian B0 (bukan V2-S ~21.5M) agar rasio parameter/sampel optimal di $N=1.826$. *Depthwise separable conv* sangat efisien membaca korelasi harmonik waktu-frekuensi spektrogram $128 \times 626$ pada 32 kHz |
| **Diversitas / Ensemble (B2)** | **ConvNeXt-Tiny** (`cv_arch_convnext_modern_convnet`) | ~28M | Kernel depthwise $7 \times 7$ menangkap modulasi frekuensi kicau Kutilang (2-4 kHz) dan sebaran broadband Bondol Jawa (2-6 kHz) [E-016]. Digunakan dengan regularisasi DropPath=0.2 sebagai kandidat *ensemble blend* |
| **Pembanding Arsitektural (B3)** | **ResNet-34 + Attention** (`cv_arch_resnet_deep_residual` & `cv_arch_cbam_channel_spatial_attention`) | ~21M | Atensi kanal & spasial menekan *noise* latar belakang hutan (hujan, serangga) [E-015]. Prioritaskan varian `timm` yang atensinya sudah *pretrained* (`seresnet34` / `eca_resnet34d`), atau pastikan modul CBAM *unfrozen* sejak Fase 1 LP-FT |
| **Opsional Audio (B4)** | **EfficientAT `mn10_as`** | ~4.8M | Hanya dieksplorasi pada 16 kHz jika membutuhkan tambahan diversitas modalitas untuk ensemble setelah B1-B3 selesai |

**Adaptasi Input & Arsitektur Bottleneck Head (Seragam untuk Semua Backbone):**

1. **Adaptasi Kanal Input:** Gunakan `in_chans=1` pada `timm.create_model(..., pretrained=True, in_chans=1)` yang merata-ratakan bobot konvolusi RGB ke 1-kanal (hemat memori dibanding duplikasi 3-kanal). Jika menggunakan duplikasi 3-kanal `.repeat(3, 1, 1)`, dilarang memakai normalisasi mean/std ImageNet karena input sudah berupa Z-score per sampel (Fase 1.6).
2. **Bottleneck Projection untuk OOD Embedding:** Dimensi fitur asli setelah Global Average Pooling (GAP) terlalu besar ($d=1280$ pada EfficientNet-B0, $d=768$ pada ConvNeXt-Tiny) untuk jumlah sampel per fold ($N \approx 1.369$), yang memicu *curse of dimensionality* pada perhitungan jarak kNN. Oleh karena itu, seluruh model menggunakan struktur *head* seragam:

```
GAP (d) --> Linear(d, 256) --> BatchNorm1d --> SiLU --> Dropout(0.3) --> Linear(256, 3)
```
- **Output Klasifikasi:** 3-class logits dari `Linear(256, 3)`.
- **Output OOD Embedding:** Vektor **256-dimensi** (sebelum Dropout dan Linear terakhir) yang di-L2 normalize untuk perhitungan jarak kNN (`k=20`). Penyeragaman ke 256-dim juga memungkinkan penggabungan jarak kNN lintas model saat ensemble.

**Protokol Adaptasi: LP-FT 2-Fase (`cv_trans_lp_ft_two_phase_adaptation`):**
- **Phase 1 (Linear Probing - 5 Epoch):** Bekukan (*freeze*) seluruh *backbone* konvolusi, latih hanya *bottleneck head* (`Linear(d, 256)` hingga `Linear(256, 3)`) dengan `lr=1e-3`. (Catatan khusus B3: jika memakai CBAM *random init*, modul CBAM wajib ikut di-*unfreeze* di Fase 1). Mencegah *feature distortion* pada bobot pretrained akibat gradien awal yang besar.
- **Phase 2 (Fine-Tuning - 20-25 Epoch):** Buka seluruh layer (*unfreeze all*), latih *end-to-end* dengan *differential learning rate* (`lr_backbone=1e-4` turun ke `1e-5`, `lr_head=5e-4`). Korpus mencatat LP-FT meningkatkan akurasi dan ketahanan OOD sebesar $+2.0\%$ hingga $+6.0\%$.

**Loss Function (Revised):**

```
loss = FocalLoss(gamma=2.0, alpha=class_balanced_weights) + LabelSmoothing(epsilon=0.05)
```

- [FIX-F] **Class-Balanced Focal Loss** menggantikan plain Label Smoothing CE. Focal Loss ($\gamma=2.0$) secara aktif menurunkan kontribusi gradien dari sampel mudah (Bondol Jawa yang dominan), memfokuskan training pada sampel sulit dan kelas minoritas.
- **Class-Balanced Effective Number Weighting** (Cui et al., CVPR 2019): $w_c = \frac{1 - \beta}{1 - \beta^{n_c}}$ dengan $\beta = 0.99$. Menghasilkan bobot yang lebih halus daripada inverse frequency untuk IR = 13.54 [E-011].
- Label smoothing diturunkan ke $\epsilon = 0.05$ (dari 0.1) agar tidak mengaburkan sinyal pembeda antar kelas yang sudah sedikit (Cendrawasih N=83).
- Ablasi wajib: Focal+CB vs plain CE+class_weight vs Focal tanpa CB.

**Optimizer & Scheduler:**

| Komponen | Spesifikasi | Sumber Korpus |
|---|---|---|
| Optimizer | AdamW, wd=1e-2 | `cv_opt_adamw_decoupled_weight_decay` |
| Scheduler | Cosine annealing with linear warmup (5% total steps) | `cv_opt_cosine_annealing_warmup` |
| Gradient clipping | Max norm 1.0 | Stabilitas training |

---

## Fase 4: Open-Set Unknown Detection

### 4.1 Arsitektur Berlapis (pada Model 3-Kelas Final)

[FIX-D] Seluruh OOD scoring dilakukan pada **model 3-kelas yang sama** yang dipakai untuk prediksi final. Tidak ada model 2-kelas terpisah. Ini menghindari masalah distribusi logit yang tidak kompatibel antar model dengan jumlah kelas berbeda.

Tiga metode OOD scoring dijalankan paralel, lalu digabung:

**Layer 1: Maximum Softmax Probability (MSP)**
```
msp_score = max(softmax(logits))
```
- Baseline sederhana. Sampel dengan `msp_score < tau_msp` diklasifikasi Unknown.
- Kelemahan: model overconfident terhadap OOD (mitigasi parsial via label smoothing).

**Layer 2: Energy Score (Liu et al., NeurIPS 2020)**
```
energy = -T * log(sum(exp(logits / T)))
```
- Temperature T di-tune pada OOF.
- Secara teori lebih baik dari MSP karena memanfaatkan distribusi seluruh logit, bukan hanya argmax.
- Semakin rendah energy, semakin kemungkinan in-distribution.

**Layer 3: kNN Distance pada 256-d Bottleneck Embedding (Sun et al., ICML 2022)**
```
knn_dist = mean(top_k_distances(embedding_256d, train_embeddings_256d, k=20))
```
- Bangun index L2-normalized **256-d bottleneck embedding** dari training set.
- Hitung jarak rata-rata ke 20-nearest neighbor.
- Sampel dengan jarak besar kemungkinan OOD.
- Sumber korpus: `cv_ood_knn_embedding`.
- **Fold-safe:** Index dibangun strictly dari train split per fold.

### 4.2 Ensemble OOD Score

```
ood_score = w1 * norm(energy_score) + w2 * norm(knn_distance)
if ood_score > tau_final:
    label = "Unknown"
else:
    label = argmax(softmax(logits))
```

Normalisasi: min-max scaling berdasarkan distribusi OOF in-distribution scores.

### 4.3 Threshold Calibration [FIX-D: Revised Protocol]

[FIX-D] **LOCO 12-model protocol DIBATALKAN.** Masalah: model 2-kelas menghasilkan distribusi logit/energy yang secara matematis tidak kompatibel dengan model 3-kelas final ($E(x) = -T \ln \sum_{i=1}^K e^{z_i/T}$ bergantung pada $K$). Melatih 12 model tambahan juga tidak efisien secara komputasi.

**Protokol pengganti: Pseudo-OOD Injection pada Model 3-Kelas**

1. Latih model 3-kelas pada 4-fold StratifiedGroupKFold (sama seperti Fase 3).
2. Kumpulkan OOF logits dan 256-d embeddings dari keempat fold (N=1826 prediksi).
3. Untuk kalibrasi threshold, gunakan **2 sumber pseudo-OOD**:
   - **Acoustic background noise:** Segmen audio train dengan silence ratio tinggi (>50%) atau RMS sangat rendah, diperlakukan sebagai proxy Unknown.
   - **Synthetic OOD via CutMix/spectral perturbation:** Campurkan potongan spektrogram dari 2 kelas berbeda dengan rasio 50:50, menghasilkan sinyal yang bukan milik kelas manapun.
4. Hitung distribusi energy score dan kNN distance pada:
   - In-distribution: OOF predictions yang benar (confident correct).
   - Pseudo-OOD: sampel dari langkah 3.
5. Pilih threshold $\tau$ yang memaksimalkan Macro F1 pada campuran in-distribution + pseudo-OOD.
6. **Fallback sederhana:** Jika pseudo-OOD injection tidak stabil, gunakan persentil ke-95 distribusi energy score in-distribution sebagai threshold. Karena `energy = -T * log(sum(exp(logits/T)))`, sampel in-distribution memiliki energy rendah (sangat negatif) dan sampel OOD memiliki energy tinggi (mendekati nol), maka klasifikasi Unknown jika `energy > percentile_95(in_distribution_energy)`.

**Keunggulan vs LOCO:**
- Tidak perlu melatih model tambahan.
- Distribusi logit/energy konsisten dengan model 3-kelas final.
- Pseudo-OOD dapat dikontrol volumenya untuk menyesuaikan sensitivity.

### 4.4 Optimasi Threshold Final

- Metode: Nelder-Mead / Powell pada OOF Macro F1 (4 kelas termasuk pseudo-Unknown).
- Sumber korpus: `validation_threshold_tuning_leakage`.
- **Constraint:** Threshold fitted strictly pada OOF, tidak pernah pada test set.

---

## Fase 5: Post-Processing & Submission

### 5.1 Temperature Scaling (Opsional)

Post-hoc calibration pada logits OOF sebelum threshold final.
- Fit skalar T pada OOF validation split.
- Sumber korpus: `cv_eval_temperature_platt_isotonic_scaling`.
- **Leakage guard:** T fitted pada validation fold, applied pada test.

### 5.2 Submission Assertions

Sebelum setiap submission, validasi otomatis:
- Jumlah baris == 814 (sama dengan test set).
- Setiap `file_id` dari `sample_submission.csv` muncul tepat sekali.
- Setiap `label` adalah salah satu dari: Bondol Jawa, Cendrawasih, Kutilang, Unknown.
- Kolom bernama `file_id` dan `label`.
- Distribusi label dicetak untuk sanity check (proporsi Unknown tidak ekstrem).

### 5.3 Bundle Protocol [D-004]

```
submissions/sub_NNN_YYYYMMDD_HHMM/
    submission.csv
    notebook.ipynb (snapshot)
    metrics.json (skor CV, detail model, threshold, skor public LB)
```

---

## Pemetaan ke Rubrik Laporan (65% Nilai Total)

| Rubrik | Bobot | Aspek yang Mendukung |
|---|---|---|
| Evaluasi & Validasi | 30% | 4-fold StratifiedGroupKFold + Pseudo-OOD Injection threshold calibration + Global OOF Macro F1 + submission assertion + ablasi augmentasi |
| Modeling | 25% | 2-track (LightGBM baseline vs ImageNet 2D CNN: EfficientNet-B0, ConvNeXt-Tiny, ResNet-34+Attention) + LP-FT 2-phase transfer + 256-d bottleneck ensemble OOD scoring |
| Preprocessing & FE | 20% | Setiap keputusan terikat evidence ID (E-xxx). Pemilihan default 32 kHz dan mel-spectrogram parameter dijustifikasi oleh analisis spektral kuantitatif |
| EDA | 15% | Cell 1-7: distribusi kelas, akustik, spektral, stereo, mel visualization. Artefak visual tersimpan |
| Penulisan | 10% | Semua artefak dan log eksperimen terstruktur di PROJECT_STATE.md |

---

## Urutan Eksekusi

| Cell | Deskripsi | Dependensi | Compute |
|---|---|---|---|
| 8 | 4-Fold StratifiedGroupKFold partition | metadata.csv | CPU |
| 9 | Preprocessing pipeline function (default 32 kHz) | Cell 8 | CPU |
| 10 | Classical ML baseline (LightGBM) | Cell 9 | CPU |
| 11 | Deep learning model setup (EfficientNet-B0 / ConvNeXt-Tiny / ResNet-34 + 256-d bottleneck) | Cell 9 | GPU |
| 12 | LP-FT 2-Phase training loop + OOF 256-d embedding collection | Cell 11 | GPU |
| 13 | OOD detection: Energy + kNN (256-d) + Pseudo-OOD Injection | Cell 12 | CPU/GPU |
| 14 | Threshold optimization (Nelder-Mead) | Cell 13 | CPU |
| 15 | Test inference + submission | All | GPU |

---

## Ablasi Wajib (untuk Rubrik Validasi 30%)

| ID | Eksperimen | Hipotesis | Metrik |
|---|---|---|---|
| ABL-01 | Dengan vs tanpa SpecAugment | SpecAugment meningkatkan generalisasi OOF | Macro F1 OOF |
| ABL-02 | Dengan vs tanpa Mixup | Mixup mengatasi imbalance Cendrawasih | F1 per-kelas Cendrawasih |
| ABL-03 | MSP saja vs Energy saja vs kNN saja vs Ensemble | Ensemble OOD scoring unggul | Macro F1 OOF (4 kelas) |
| ABL-04 | LP-FT vs Direct fine-tune | LP-FT mempertahankan fitur OOD-aware (+2-6% OOD) | Macro F1 OOF + AUROC OOD |
| ABL-05 | 32 kHz (default) vs 16 kHz | 32 kHz meningkatkan F1 Cendrawasih pada EfficientNet-B0 karena menangkap harmonik 8-16 kHz (9.08% energi) [E-017] | F1 Cendrawasih + Macro F1 OOF |
| ABL-06 | Label smoothing 0.05 vs 0.0 | Label smoothing memperbaiki kalibrasi OOD | ECE + Macro F1 OOF |
| ABL-07 | Focal+CB vs plain CE+class_weight | Class-Balanced Focal Loss unggul pada kelas minoritas | F1 Cendrawasih + Macro F1 OOF |
| ABL-08 | reflect-pad vs wrap-around vs zero-pad | Metode padding mempengaruhi performa pada audio pendek | Macro F1 OOF pada subset durasi < 8s |

---

## Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Pretrained weights dianggap data eksternal oleh panitia | Diskualifikasi | Menggunakan ImageNet weights (domain visual, bukan data audio eksternal) [A-004] dan menyiapkan training from scratch sebagai fallback. Lampirkan identitas model di notebook |
| Cendrawasih hanya 83 sampel, fold validasi bisa berisi 3-35 sampel | Metrik per-fold tidak stabil | Gunakan Global OOF Macro F1 (gabungan 1826 sampel) [FIX-A]. Gunakan backbone ringan EfficientNet-B0 (~5.3M) + Mixup + Class-Balanced Focal Loss [D-007] |
| Threshold Unknown tidak terkalibrasi (tanpa ground truth) | Macro F1 Unknown rendah | Pseudo-OOD Injection pada model 3-kelas + Nelder-Mead optimization pada OOF [FIX-D] |
| Curse of dimensionality pada ruang embedding kNN OOD | Jarak kNN tidak diskriminatif pada $d=1280$ | Reduksi dimensi fitur ke 256-d melalui bottleneck projection layer + L2 normalization wajib (`k=20`) [D-007] |

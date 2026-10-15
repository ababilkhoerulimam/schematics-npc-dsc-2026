# Preprocessing & Methodology Plan

**Project:** Schematics NPC DSC 2026 - Bird Chirp Openset Classification
**Version:** 1.0
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
- **Negligible ultrasonic content:** Band 16-24 kHz hanya 0.11-1.22% energi total [E-017]

---

## Fase 0: Partisi Data

**Metode:** 4-fold StratifiedGroupKFold pada `source_group`, stratified by `label`.

**Justifikasi:** Cendrawasih dan Kutilang masing-masing punya tepat 4 source groups [E-010]. 4-fold membagi tepat 1 group per fold per kelas minoritas dengan zero recording-source leakage [D-001].

**Output:** `metadata_with_folds.csv` dengan kolom `fold` (0-3).

**Validasi:**
- Setiap fold memuat representasi ketiga kelas.
- Zero group overlap antar train-val split.
- Distribusi label per fold proporsional.

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
| Target SR (pretrained backbone) | 16.000 Hz | Standar input AST/BEATs/EfficientAT |
| Target SR (custom CNN) | 32.000 Hz | Mempertahankan harmonik 8-16 kHz (hingga 9.08% energi Cendrawasih) [E-017, D-005] |
| Metode | `librosa.resample` (kaiser_best) | Anti-aliasing filter bawaan |

`target_sr` adalah parameter pipeline, bukan hardcoded. Default 16 kHz saat menggunakan pretrained backbone.

### 1.3 Peak Normalization

```
x_norm = x / max(|x|)
```

| Parameter | Nilai | Justifikasi |
|---|---|---|
| Method | Per-sample peak normalization ke [-1, 1] | 40-55% rekaman mengalami clipping digital >= 0.99 [E-015]. Normalisasi menyeragamkan dynamic range tanpa mengubah distribusi spektral relatif |
| Alternatif ditolak | Soft peak -1.0 dBFS | Membutuhkan konversi dB yang menambah kompleksitas tanpa keuntungan terukur |

### 1.4 Standardisasi Durasi (10.0 Detik)

| Kondisi | Strategi | Justifikasi |
|---|---|---|
| Audio < 10s | Reflect-pad (memantulkan sinyal) | Durasi minimum 5.27s [E-009]. Reflect-pad menghindari artefak periodesitas palsu yang muncul pada wrap-around repeat dan menghindari injeksi silence pada zero-pad |
| Audio > 10s (train) | Random crop 10s | Augmentasi implisit posisi temporal |
| Audio > 10s (test) | Center crop 10s | Deterministik dan reproducible |

Jumlah sampel pada 16 kHz: 160.000. Pada 32 kHz: 320.000.

### 1.5 Log-Mel Spectrogram

| Parameter | Nilai (16 kHz) | Nilai (32 kHz) | Justifikasi |
|---|---|---|---|
| n_fft | 1024 (~64 ms) | 2048 (~64 ms) | Resolusi frekuensi cukup untuk memisahkan band 2-4 kHz Kutilang dari 4-6 kHz Bondol [E-016] |
| hop_length | 320 (~20 ms) | 512 (~16 ms) | 75% overlap, resolusi temporal memadai untuk pola repetitif kicauan |
| n_mels | 128 | 128 | Standar untuk audio classification, resolusi mel cukup untuk 3 kelas dengan signature spektral berbeda |
| fmin | 200 Hz | 200 Hz | Memotong low-frequency rumble (angin, getaran tanah, hum sensor) yang tidak informatif |
| fmax | 8000 Hz | 16000 Hz | Disesuaikan Nyquist; mempertahankan seluruh band informatif [E-017] |
| Konversi dB | power_to_db, ref=max, top_db=80 | Idem | Kompresi range dinamis untuk representasi visual dan input neural network |

**Dimensi output (16 kHz):** 128 x 501 (128 mel bins x ceil(160000/320) time frames).

### 1.6 Per-Sample Normalization

```
S_norm = (S_dB - mean(S_dB)) / (std(S_dB) + 1e-6)
```

Normalisasi z-score per spektrogram individual. Meredam variasi gain mikrofon lapangan tanpa memerlukan statistik global dataset (fold-safe by design).

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

**Validasi:** 4-fold GroupKFold OOF, metrik Macro F1.

### Track B: Deep Learning (GPU)

**Option B1: Pretrained Audio Transformer (Rekomendasi Utama)**

| Aspek | Spesifikasi |
|---|---|
| Kandidat backbone | EfficientAT (mn10_as) / AST (MIT) / BEATs |
| Input | Log-Mel spectrogram 128 x T pada 16 kHz |
| Transfer protocol | LP-FT 2-phase (corpus: `cv_trans_lp_ft_two_phase_adaptation`) |
| Phase 1 (LP) | Freeze backbone, train linear head, 5-10 epoch, lr=1e-3 |
| Phase 2 (FT) | Unfreeze all, fine-tune end-to-end, lr=1e-5, 20-30 epoch |
| Output | 3-class softmax logits + penultimate layer embedding (untuk OOD detection) |
| Catatan | Weights pretrained pada AudioSet = parameter model, bukan data eksternal [A-004] |

**Option B2: Custom Lightweight CNN (Fallback)**

| Aspek | Spesifikasi |
|---|---|
| Arsitektur | 4-5 conv blocks (Conv2d-BN-ReLU-MaxPool) |
| Input | Log-Mel spectrogram 128 x T pada 32 kHz |
| Pooling | Global Average Pooling ke 256-dim embedding |
| Head | Linear 256 ke 3 kelas |
| Kapan dipakai | Jika pretrained backbone dilarang atau terlalu berat untuk budget komputasi |

**Loss Function:**

```
loss = LabelSmoothingCrossEntropy(epsilon=0.1, weight=class_weights)
```

- Label smoothing menurunkan overconfidence (corpus: `cv_opt_label_smoothing_calibration`), krusial agar distribusi softmax probability lebih terkalibrasi untuk OOD detection di Fase 4.
- `class_weights` inversely proportional to class frequency: mengatasi IR = 13.54 [E-011].

**Optimizer & Scheduler:**

| Komponen | Spesifikasi | Sumber Korpus |
|---|---|---|
| Optimizer | AdamW, wd=1e-2 | `cv_opt_adamw_decoupled_weight_decay` |
| Scheduler | Cosine annealing with linear warmup (5% total steps) | `cv_opt_cosine_annealing_warmup` |
| Gradient clipping | Max norm 1.0 | Stabilitas training |

---

## Fase 4: Open-Set Unknown Detection

### 4.1 Arsitektur Berlapis

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

**Layer 3: kNN Distance pada Embedding (Sun et al., ICML 2022)**
```
knn_dist = mean(top_k_distances(embedding, train_embeddings, k=20))
```
- Bangun index L2-normalized penultimate embedding dari training set.
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

### 4.3 Threshold Calibration: Leave-One-Class-Out (LOCO)

Karena kelas Unknown tidak punya label di train, kalibrasi threshold dilakukan dengan LOCO:

1. Pada setiap fold, tahan satu kelas known sebagai pseudo-Unknown.
2. Train model pada 2 kelas remaining.
3. Evaluasi: apakah OOD detector menolak kelas yang ditahan?
4. **3 rotasi** (tahan Bondol, tahan Kutilang, tahan Cendrawasih) x **4 fold** = 12 skenario.
5. Pilih threshold yang memaksimalkan rata-rata Macro F1 di semua 12 skenario.

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
| Evaluasi & Validasi | 30% | 4-fold StratifiedGroupKFold + LOCO pseudo-OOD + OOF threshold optimization + submission assertion + ablasi augmentasi |
| Modeling | 25% | 2-track (classical baseline vs deep learning) + LP-FT transfer + ensemble OOD scoring berlapis |
| Preprocessing & FE | 20% | Setiap keputusan terikat evidence ID (E-xxx). Mel-spectrogram parameter dijustifikasi oleh analisis spektral kuantitatif |
| EDA | 15% | Cell 1-7: distribusi kelas, akustik, spektral, stereo, mel visualization. Artefak visual tersimpan |
| Penulisan | 10% | Semua artefak dan log eksperimen terstruktur di PROJECT_STATE.md |

---

## Urutan Eksekusi

| Cell | Deskripsi | Dependensi | Compute |
|---|---|---|---|
| 8 | 4-Fold StratifiedGroupKFold partition | metadata.csv | CPU |
| 9 | Preprocessing pipeline function | Cell 8 | CPU |
| 10 | Classical ML baseline (LightGBM) | Cell 9 | CPU |
| 11 | Deep learning model setup | Cell 9 | GPU |
| 12 | Training loop + OOF embedding collection | Cell 11 | GPU |
| 13 | OOD detection: Energy + kNN + LOCO | Cell 12 | CPU/GPU |
| 14 | Threshold optimization (Nelder-Mead) | Cell 13 | CPU |
| 15 | Test inference + submission | All | GPU |

---

## Ablasi Wajib (untuk Rubrik Validasi 30%)

| ID | Eksperimen | Hipotesis | Metrik |
|---|---|---|---|
| ABL-01 | Dengan vs tanpa SpecAugment | SpecAugment meningkatkan generalisasi OOF | Macro F1 OOF |
| ABL-02 | Dengan vs tanpa Mixup | Mixup mengatasi imbalance Cendrawasih | F1 per-kelas Cendrawasih |
| ABL-03 | MSP saja vs Energy saja vs kNN saja vs Ensemble | Ensemble OOD scoring unggul | Macro F1 OOF (4 kelas) |
| ABL-04 | LP-FT vs Direct fine-tune | LP-FT mempertahankan fitur OOD-aware | Macro F1 OOF + AUROC OOD |
| ABL-05 | 16 kHz vs 32 kHz | 32 kHz tidak memberikan keuntungan signifikan pada pretrained backbone | Macro F1 OOF |
| ABL-06 | Label smoothing 0.1 vs 0.0 | Label smoothing memperbaiki kalibrasi OOD | ECE + Macro F1 OOF |

---

## Risiko dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Pretrained weights dianggap data eksternal oleh panitia | Diskualifikasi | Siapkan Track B2 (custom CNN from scratch) sebagai fallback. Lampirkan identitas model di notebook [Constraint kompetisi] |
| Cendrawasih hanya 83 sampel, fold validasi bisa berisi 3-35 sampel | Metrik per-fold tidak stabil | Gunakan rata-rata 4-fold dan monitor variance. Mixup + class weight sebagai mitigasi |
| Threshold Unknown tidak terkalibrasi (tanpa ground truth) | Macro F1 Unknown rendah | LOCO protocol 12-skenario + Nelder-Mead optimization pada OOF |
| Overfitting pada embedding space kecil | kNN OOD detector gagal generalisasi | L2 normalization wajib, k=20 memberikan smoothing lokal |

# Execution Plan: Day 4 Lean Experimentation (Transductive Self-Training, Free-Energy OOD, & Multi-Submission)

**Proyek:** Schematics NPC DSC 2026 (Kompetisi Sains Data Nasional ITS)
**Tugas:** Audio Open-Set Classification (Bondol Jawa, Cendrawasih, Kutilang, dan Unknown)
**Target:** Menembus batas Public Leaderboard dari 0.66760 (sub_013) menuju 0.7200 - 0.7600
**Dokumen:** `planning_day_4_experiment.md`
**Versi:** 1.0 (Lean 17-Cell Execution Blueprint: Zero EDA Overhead, In-Fold Pseudo-Labeling, & Tri-Submission Export)
**Status:** Approved & Ready for Interactive Execution

---

## 1. Filosofi & Karakteristik Rencana Harian

Dokumen ini disusun khusus sebagai **panduan eksekusi teknis harian** (bukan laporan naratif panjang). Seluruh tahapan eksplorasi data (EDA, Kruskal-Wallis, cek korelasi stereo, dsb.) dieliminasi karena telah terbukti secara permanen pada Day 1 s.d. Day 3.

### 4 Prinsip Kunci Eksekusi:
1. **Lean & Rapid:** Hanya 17 sel terarah. Total estimasi waktu eksekusi dari Cell 1 hingga Cell 17 adalah $\approx 12\text{--}15\text{ menit}$ di Kaggle GPU Tesla T4.
2. **Zero External File Upload:** 100% *self-contained*. Menggunakan bobot pretrained publik resmi (`perch_v2/2` dari TFHub dan package `birdnet` dari PyPI).
3. **Strict VRAM Isolation:** Ekstraksi backbone berat dilakukan di awal, segera dibersihkan dari memori via `purge_vram()`, lalu GPU dialokasikan penuh untuk pelatihan head PyTorch.
4. **Anti-Shakeup Multi-Submission:** Mengalokasikan kuota submission harian ke dalam 3 profil risiko berbeda (`sub_014` Anchor, `sub_015` SOTA Champion, dan `sub_016` Conservative Probe).

---

## 2. Struktur 5 Tahap Pipeline Eksekusi (17 Sel)

```
[Tahap 1: Environment & DSP Pre-Caching] (Sel 1 - 3)
  ├── Cell 1: Environment Setup, Seed 42, TF Memory Growth, Purge VRAM Helper.
  ├── Cell 2: Remote Dataset Discovery & Load Metadata with Locked Folds.
  └── Cell 3: Fast Dual-Stream DSP RAM Pre-Caching (32 kHz 5s & 48 kHz 3s).

[Tahap 2: Foundation Feature Extraction] (Sel 4 - 6)
  ├── Cell 4: Batch Extraction Google Perch v2 (Train & Test Embedding 1536-d + Logit Summary).
  ├── Cell 5: Batch Extraction Official BirdNET v2.4 (Logit 4-d: Bondol, Kutilang, Cendra, NonTarget).
  └── Cell 6: Purge VRAM & Memory Sanity Check.

[Tahap 3: Supervised Baseline & In-Fold Thresholding] (Sel 7 - 9)
  ├── Cell 7: In-Fold AUROC Taxa Verification & Training PerchBottleneckCosineHead (4 Folds).
  ├── Cell 8: In-Fold Youden Knee-Point Threshold Selection (Cendrawasih & Trucuk).
  └── Cell 9: Baseline OOF Macro F1 Evaluation & Initial Closed-Set Probability Matrix.

[Tahap 4: Day 4 Core Innovation - Transductive Self-Training] (Sel 10 - 13)
  ├── Cell 10: Helmholtz Free-Energy Scoring E(x) & Adaptive Test Inlierness Curation (N=120-160).
  ├── Cell 11: In-Fold Augmented Dataset Construction (Train + Pseudo-Test per Fold).
  ├── Cell 12: Retraining 4-Fold Bottleneck Cosine Heads (Self-Training 40 Epochs per Fold).
  └── Cell 13: Self-Trained Test Inference & Test Probability Matrix.

[Tahap 5: Forensic Shielding & Tri-Submission Export] (Sel 14 - 17)
  ├── Cell 14: Forensic Cendrawasih Reassignment (thr Knee-Point) + Trucuk Isolation.
  ├── Cell 15: Sweet-Spot Manifold Unknown Quantile Locking (~43.5%).
  ├── Cell 16: Assembly & Export Tri-Submission (sub_014, sub_015, sub_016).
  └── Cell 17: Zip Working Directory Artifacts & Display Direct Download Link.
```

---

## 3. Rincian Sel-demi-Sel & Expected Verified Outputs

### Tahap 1: Environment & DSP Pre-Caching (Sel 1 - 3)

* **Cell 1: Environment Setup, Seed 42, TF Memory Growth, and Purge VRAM Helper**
  * *Operasi:* Inisialisasi deterministik seed 42 (PyTorch, CuDNN, NumPy). Aktifkan TensorFlow dynamic memory growth (`set_memory_growth=True`) untuk mencegah monopoli VRAM. Definisikan helper `purge_vram()`.
  * *Input:* Runtime hardware.
  * *Expected Output:* `hardware_target | device=cuda:0 | gpu=Tesla T4 | vram_free=14.46GB | tf_memory_growth=True`.

* **Cell 2: Remote Dataset Discovery & Load Metadata with Locked Folds**
  * *Operasi:* Autodeteksi base directory `/kaggle/input/competitions/schematics-2026`. Muat `metadata_with_folds.csv` (1.826 train) dan `sample_submission.csv` (814 test). Assert partisi fold `[35, 29, 16, 3]` untuk Cendrawasih.
  * *Input:* `metadata_with_folds.csv`, `sample_submission.csv`.
  * *Expected Output:* `metadata_loaded | train=1826 | test=814 | fold_cendra=[35, 29, 16, 3] | group_leakage=0`.

* **Cell 3: Fast Dual-Stream DSP RAM Pre-Caching (32 kHz 5s & 48 kHz 3s)**
  * *Operasi:* Pre-caching audio ke memori RAM:
    1. Jalur Perch (32 kHz): Resampling polyphase, High-Pass 100 Hz Butterworth, Linear Peak Norm 0.95, Dynamic Equidistant 3-Window 5.0s (160.000 sampel).
    2. Jalur BirdNET (48 kHz): Downmix mono, Dynamic 3-Window 3.0s (144.000 sampel).
  * *Input:* Raw audio files.
  * *Expected Output:* `ram_caching_complete | train_32k=1826 | test_32k=814 | train_48k=1826 | test_48k=814 | elapsed ~70s`.

---

### Tahap 2: Foundation Feature Extraction (Sel 4 - 6)

* **Cell 4: Batch Extraction Google Perch v2 (Train & Test Embedding 1536-d + Logit Summary)**
  * *Operasi:* Muat model `perch_v2/2` dari TFHub/Kaggle Models. Ekstraksi batch 32 menghasilkan embedding L2-norm 1536-d dan ringkasan logit kandidat 16-d untuk 1.826 train dan 814 test.
  * *Input:* Pre-cached audio 32 kHz.
  * *Expected Output:* `perch_extracted | train_emb=(1826, 3, 1536) | test_emb=(814, 3, 1536) | elapsed ~35s`.

* **Cell 5: Batch Extraction Official BirdNET v2.4 (Logit 4-d Summary)**
  * *Operasi:* Inisialisasi pustaka resmi `birdnet` v1.3.0 LiteRT backend. Ekstraksi logit 3-window 48 kHz dengan checkpointing chunk (CHUNK_FILES=64) menghasilkan matriks ringkasan 4-d: `[Bondol, Kutilang, MaxCendra (17 genus), MaxNonTarget]`.
  * *Input:* Pre-cached audio 48 kHz.
  * *Expected Output:* `birdnet_extracted | train_sum=(1826, 3, 4) | test_sum=(814, 3, 4) | elapsed ~9m | nan=0`.

* **Cell 6: Purge VRAM & Memory Sanity Check**
  * *Operasi:* Hapus objek model Perch dan BirdNET dari Python context, jalankan `purge_vram()`, dan verifikasi VRAM GPU kembali bebas >13 GB sebelum melatih neural network PyTorch.
  * *Input:* System state.
  * *Expected Output:* `vram_purged | vram_free > 13.5GB | ram_used < 7GB`.

---

### Tahap 3: Supervised Baseline & In-Fold Thresholding (Sel 7 - 9)

* **Cell 7: In-Fold AUROC Taxa Verification & Training PerchBottleneckCosineHead (4 Folds)**
  * *Operasi:* Seleksi taksa Cendrawasih murni in-fold (mengunci indeks 2893). Latih `PerchBottleneckCosineHead` (`Linear(1536, 256) -> LayerNorm -> SiLU -> Dropout(0.2) -> CosineLinear(256, 3)`) selama 40 epoch per fold dengan MIL Softmax Bag Pooling. Evaluasi wajib pada epoch terakhir 40 (zero validation peeking).
  * *Input:* `train_emb`, `metadata_with_folds.csv`.
  * *Expected Output:* `baseline_head_trained | Global OOF Macro F1 > 0.9280 | Bondol > 0.98 | Kutilang > 0.99 | Cendra > 0.80 | elapsed ~35s`.

* **Cell 8: Nested In-Fold Knee-Point Threshold Selection (Cendrawasih & Trucuk)**
  * *Operasi:* Tentukan titik infleksi (*knee point* / Maximum Youden Index) threshold BirdNET per fold:
    * `thr_cendra` (titik optimasi reassignment Cendrawasih $\approx -7.70$).
    * `thr_kutilang` (titik detektor Kutilang mati $\approx -8.11$).
  * *Input:* `p_perch_oof`, `train_birdnet_sum`.
  * *Expected Output:* `nested_knee_points_locked | thr_cendra mean ~ -7.70 | thr_kutilang mean ~ -8.11`.

* **Cell 9: Baseline OOF Macro F1 Evaluation & Initial Closed-Set Probability Matrix**
  * *Operasi:* Evaluasi OOF dengan menerapkan soft reassignment Cendrawasih ke pilihan kedua Perch. Ekstraksi probabilitas inferensi awal pada 814 test set (`p_perch_test`).
  * *Input:* Model 4 fold dari Cell 7, threshold dari Cell 8.
  * *Expected Output:* `reassigned_oof_macro_f1 > 0.9450 | test_p_perch shape=(814, 3)`.

---

### Tahap 4: Day 4 Core Innovation - Transductive Self-Training (Sel 10 - 13)

* **Cell 10: Helmholtz Free-Energy Scoring E(x) & Adaptive Test Inlierness Curation**
  * *Operasi:* Hitung skor energi bebas $E(x) = -T \log \sum_{c=0}^2 \exp(z_c / T)$ dan jarak manifold kNN-5 mean-centered. Seleksi adaptif $\approx 120\text{--}150$ sampel test paling bersih menggunakan 3 lapis filter konsensus (Top 25% confidence + Top 40% manifold distance + BirdNET verification).
  * *Input:* `p_perch_test`, `test_d_perch`, `test_bnet_sum`.
  * *Expected Output:* `curated_pseudo_samples: N in [120, 160] | zero_unknown_overlap=True`.

* **Cell 11: In-Fold Augmented Dataset Construction (Train + Pseudo-Test per Fold)**
  * *Operasi:* Alokasikan pseudo-samples ke porsi train di setiap fold (menjaga data validasi lokal tetap murni 100% held-out in-distribution). Hitung bobot inverse-frequency class weights untuk mencegah dominasi kelas mayoritas.
  * *Input:* Fold train splits asli + curated pseudo-test.
  * *Expected Output:* `in_fold_augmented_ready | fold_train_sizes ~1460 + N_pseudo | fold_val_sizes ~450`.

* **Cell 12: Retraining 4-Fold Bottleneck Cosine Heads (Self-Training 40 Epochs per Fold)**
  * *Operasi:* Latih ulang 4 model head klasifikasi pada augmented manifold menggunakan Cosine Annealing LR selama 40 epoch per fold.
  * *Input:* Augmented datasets dari Cell 11.
  * *Expected Output:* `self_training_completed | 4 folds evaluated | total_elapsed < 45s | final_loss < 0.05`.

* **Cell 13: Self-Trained Test Inference & Test Probability Matrix**
  * *Operasi:* Forward pass ensemble 4 fold hasil self-training pada 814 data uji menghasilkan matriks probabilitas teradaptasi domain (`p_selftrained_test`).
  * *Input:* 4 model self-trained dari Cell 12, `test_emb`.
  * *Expected Output:* `p_selftrained_test shape=(814, 3) | min_sum=1.0 | max_sum=1.0`.

---

### Tahap 5: Forensic Shielding & Tri-Submission Export (Sel 14 - 17)

* **Cell 14: Forensic Cendrawasih Reassignment & Trucuk Isolation**
  * *Operasi:* Terapkan logika forensik taksonomi ganda:
    1. Cendrawasih Fallback: Jika pred == Cendrawasih dan logit BirdNET $< \text{thr\_cendra}$, kembalikan ke pilihan kedua Perch.
    2. Trucuk Shield: Jika pred == Kutilang dengan confidence marjinal ($P < 0.60$) dan logit BirdNET Kutilang $< \text{thr\_kutilang}$, isolasi ke `Unknown` (*Pycnonotus goiavier*).
  * *Input:* `p_selftrained_test`, `test_bnet_sum`.
  * *Expected Output:* `forensic_applied | cendra_reassigned ~85-95 | trucuk_shielded ~70-75`.

* **Cell 15: Sweet-Spot Manifold Unknown Quantile Locking (~43.5%)**
  * *Operasi:* Kunci ambang batas $\tau$ sedemikian rupa sehingga total Unknown (gabungan manifold distance + Trucuk shield) berada tepat di rentang optimal 43.0% - 44.0%.
  * *Input:* `test_d_perch`, `trucuk_mask`.
  * *Expected Output:* `tau_locked | total_unknown_count ~350-357 (43.0% - 43.8%)`.

* **Cell 16: Assembly & Export Tri-Submission (sub_014, sub_015, sub_016)**
  * *Operasi:* Rakit dan ekspor 3 berkas submission resmi:
    1. `experiment 14.csv` (`sub_014` - Safe Anchor): Baseline Master Ensemble tanpa self-training @ 43.5% Unknown (Anti-Shakeup Safe Bet).
    2. `experiment 15.csv` (`sub_015` - SOTA Champion): Self-Trained Perch + Forensic Shield @ 43.5% Unknown (Target 0.70+).
    3. `experiment 16.csv` (`sub_016` - Conservative Probe): Self-Trained Perch @ 41.5% Unknown.
    * Verifikasi: 814 baris, 0 null, 4 kelas valid.
  * *Input:* Label prediksi dari Cell 14-15.
  * *Expected Output:* `experiment 14, 15, 16 exported successfully | 814 rows verified`.

* **Cell 17: Zip Working Directory Artifacts & Display Direct Download Link**
  * *Operasi:* Kompres seluruh isi `/kaggle/working` menjadi `schematics_day4_all_artifacts.zip` dan tampilkan `FileLink` interaktif untuk pengunduhan instan.
  * *Input:* Seluruh file di `/kaggle/working`.
  * *Expected Output:* `schematics_day4_all_artifacts.zip created | direct download link displayed`.

---

## 4. Matriks Validasi Cepat & Keamanan Eksekusi

| Checkpoint | Yang Diverifikasi | Kriteria Lolos (Assertion) | Tindakan Jika Gagal |
|---|---|---|---|
| **Pasca Cell 3** | Kepadatan audio RAM | `zero_pad_corruption == 0.0%` | Periksa fungsi dynamic equidistant slicing. |
| **Pasca Cell 5** | Kelengkapan BirdNET | `np.isnan(test_sum).sum() == 0` | Ulangi chunk yang gagal via checkpoint cache. |
| **Pasca Cell 6** | Ketersediaan VRAM | `vram_free > 13.0 GB` | Panggil paksa `torch.cuda.empty_cache()` & restart kernel jika perlu. |
| **Pasca Cell 10** | Kurasi Pseudo-Labels | `100 <= len(pseudo_indices) <= 180` | Sesuaikan kuantil p75/p40 secara adaptif. |
| **Pasca Cell 16** | Integritas File CSV | 814 baris, 0 null, 4 kelas unik | Tolak ekspor, audit pemetaan string label. |

Blueprint eksperimen 17 sel ini disetujui, terkunci di `schematics_day4_all_artifacts/planning_day_4_experiment.md`, dan siap dieksekusi sel-demi-sel secara interaktif.

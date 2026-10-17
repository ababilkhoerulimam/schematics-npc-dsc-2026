# Execution Plan: Day 4 Lean Experimentation (Transductive Self-Training, Free-Energy OOD, & Multi-Submission)

**Proyek:** Schematics NPC DSC 2026 (Kompetisi Sains Data Nasional ITS)
**Tugas:** Audio Open-Set Classification (Bondol Jawa, Cendrawasih, Kutilang, dan Unknown)
**Target:** Menembus batas Public Leaderboard dari 0.66760 (sub_013) menuju 0.7200 - 0.7600
**Dokumen:** `planning_day_4_experiment.md`
**Versi:** 2.0 (10/10 Flawless 17-Cell Execution Blueprint: Self-Contained Combinatorial Folds, 1543-d Hybrid Input, In-Fold Pseudo-Labeling, & Pure A/B Tri-Submission)
**Status:** Approved & Locked for Interactive Execution

---

## 1. Filosofi & Karakteristik Rencana Harian

Dokumen ini disusun khusus sebagai **panduan eksekusi teknis harian** yang ramping dan bebas *bug*. Seluruh tahapan eksplorasi data (EDA, Kruskal-Wallis, cek korelasi stereo, dsb.) dieliminasi karena telah tervalidasi secara permanen pada Day 1 s.d. Day 3.

### 4 Prinsip Kunci Eksekusi (Resolusi 10/10):
1. **Lean & Rapid:** Hanya 17 sel terarah. Total estimasi waktu eksekusi dari Cell 1 hingga Cell 17 adalah $\approx 12\text{--}15\text{ menit}$ di Kaggle GPU Tesla T4.
2. **True Zero External File Upload (100% Self-Contained):** Tidak bergantung pada unggahan file CSV/NPY dari hari sebelumnya. Jika `metadata_with_folds.csv` tidak tersedia di direktori kerja, Cell 2 secara otomatis mengeksekusi algoritma *Exact Combinatorial 4-Fold StratifiedGroupKFold* langsung dari `train/metadata.csv` bawaan kompetisi.
3. **1543-d Hybrid Representation & Strict VRAM Isolation:** Menggabungkan *embedding* semantik `1536-d` dengan *verified taxonomic logit summary* `7-d` (`1536 + 7 = 1543-d`) sebagai input terpadu ke `PerchBottleneckCosineHead`. Ekstraksi *backbone* berat segera dibersihkan dari memori via `purge_vram()` di Cell 6 sebelum pelatihan PyTorch.
4. **Pure A/B Testing & Anti-Shakeup Multi-Submission:** Ketiga submission harian dirancang konsisten di atas fondasi fitur yang sama untuk pengujian hipotesis yang murni:
   * `sub_014` (*Safe Anchor*): **Pure Supervised Perch Hybrid (1543-d) + Forensic Shield @ 43.5% Unknown** (Tanpa Self-Training - pengukur baseline murni & jaring pengaman *Private LB*).
   * `sub_015` (*SOTA Champion*): **In-Fold Self-Trained Perch Hybrid (1543-d) + Forensic Shield @ 43.5% Unknown** (Mengukur dampak murni *Transductive Self-Training* secara *apple-to-apple* terhadap `sub_014`).
   * `sub_016` (*Conservative Sweet-Spot Probe*): **In-Fold Self-Trained Perch Hybrid (1543-d) + Forensic Shield @ 41.5% Unknown** (Menguji sensitivitas batas kuantil *Unknown*).

---

## 2. Struktur 5 Tahap Pipeline Eksekusi (17 Sel)

```
[Tahap 1: Environment, Self-Contained Fold Locking, & DSP Pre-Caching] (Sel 1 - 3)
├── Cell 1: Environment Setup, Seed 42, TF Memory Growth, Purge VRAM Helper.
├── Cell 2: Dataset Discovery & Self-Contained Combinatorial 4-Fold Generator.
└── Cell 3: Fast Dual-Stream DSP RAM Pre-Caching (32 kHz 5s & 48 kHz 3s).

[Tahap 2: Foundation Feature Extraction & VRAM Isolation] (Sel 4 - 6)
├── Cell 4: Batch Extraction Google Perch v2 (Embedding 1536-d + Candidate Logits 16-d).
├── Cell 5: Batch Extraction Official BirdNET v2.4 (Logit 4-d: Bondol, Kutilang, Cendra, NonTarget).
└── Cell 6: Explicit Model Deletion, Subprocess/Context VRAM Purge, & Memory Sanity Check.

[Tahap 3: Supervised Hybrid Baseline & In-Fold Knee-Point Thresholding] (Sel 7 - 9)
├── Cell 7: In-Fold Taxa Verification (Lock 7-d Summary -> 1543-d Hybrid Input) & Train Supervised Head.
├── Cell 8: Nested In-Fold Youden Knee-Point Threshold Selection (Cendrawasih & Trucuk).
└── Cell 9: Supervised Baseline OOF Evaluation & Anchor Test Probability Matrix (p_supervised_test).

[Tahap 4: Day 4 Core Innovation - Transductive In-Fold Self-Training] (Sel 10 - 13)
├── Cell 10: Helmholtz Free-Energy Scoring E(x), kNN-5 Distance, & Consensus Inlierness Curation.
├── Cell 11: In-Fold Augmented Dataset Construction (Train 1543-d + Pseudo-Test per Fold).
├── Cell 12: Retraining 4-Fold Bottleneck Cosine Heads (Self-Training 40 Epochs on 1543-d).
└── Cell 13: Self-Trained Test Inference & Adapted Probability Matrix (p_selftrained_test).

[Tahap 5: Dual Forensic Shielding & Tri-Submission Export] (Sel 14 - 17)
├── Cell 14: Dual Forensic Taxonomic Shield (Applied to both Supervised & Self-Trained Matrices).
├── Cell 15: Sweet-Spot Manifold Unknown Quantile Locking (43.5% Standard & 41.5% Probe).
├── Cell 16: Assembly, Verification, & Export Tri-Submission (sub_014, sub_015, sub_016).
└── Cell 17: Zip Working Directory Artifacts & Display Direct Download Link.
```

---

## 3. Rincian Sel-demi-Sel & Expected Verified Outputs

### Tahap 1: Environment, Self-Contained Fold Locking, & DSP Pre-Caching (Sel 1 - 3)

* **Cell 1: Environment Setup, Seed 42, TF Memory Growth, and Purge VRAM Helper**
  * *Operasi:* Inisialisasi deterministik `seed=42` (PyTorch, CuDNN, NumPy, Python Hash). Aktifkan `tf.config.experimental.set_memory_growth(gpu, True)` untuk mencegah monopoli VRAM oleh TensorFlow. Definisikan helper `purge_vram()`.
  * *Input:* Runtime hardware.
  * *Expected Output:* `hardware_target | device=cuda:0 | gpu=Tesla T4 | vram_free=14.46GB | tf_memory_growth=True`.

* **Cell 2: Remote Dataset Discovery & Self-Contained Combinatorial 4-Fold Generator**
  * *Operasi:* Autodeteksi base directory `/kaggle/input/competitions/schematics-2026`. Muat `train/metadata.csv` (1.826 baris) dan `sample_submission.csv` (814 baris). Jika `metadata_with_folds.csv` belum ada di `/kaggle/working`, jalankan algoritma *Exact Combinatorial 4-Fold Group Partitioning* yang memetakan 15 `source_group` ke 4 fold tanpa kebocoran grup dan menyimpannya ke disk. Assert distribusi Cendrawasih `[35, 29, 16, 3]`.
  * *Input:* `train/metadata.csv`, `sample_submission.csv`.
  * *Expected Output:* `metadata_locked | train=1826 | test=814 | fold_cendra=[35, 29, 16, 3] | group_leakage=0`.

* **Cell 3: Fast Dual-Stream DSP RAM Pre-Caching (32 kHz 5s & 48 kHz 3s)**
  * *Operasi:* Pre-caching audio ke memori RAM:
    1. Jalur Perch (32 kHz): Resampling polyphase, High-Pass 100 Hz Butterworth, Linear Peak Norm 0.95 (tanpa distorsi tanh), Dynamic Equidistant 3-Window 5.0s (160.000 sampel).
    2. Jalur BirdNET (48 kHz): Downmix mono lossless, Dynamic Equidistant 3-Window 3.0s (144.000 sampel).
  * *Input:* Raw audio files.
  * *Expected Output:* `ram_caching_complete | train_32k=1826 | test_32k=814 | train_48k=1826 | test_48k=814 | zero_pad_corruption=0.0% | elapsed ~70s`.

---

### Tahap 2: Foundation Feature Extraction & VRAM Isolation (Sel 4 - 6)

* **Cell 4: Batch Extraction Google Perch v2 (Embedding 1536-d + Candidate Logits 16-d)**
  * *Operasi:* Muat model `perch_v2/2` dari TFHub/Kaggle Models. Ekstraksi batch 32 menghasilkan embedding L2-normalized `(N, 3, 1536)` dan ringkasan logit kandidat taksa `(N, 3, 16)` untuk 1.826 train dan 814 test. Simpan ke `.npy`.
  * *Input:* Pre-cached audio 32 kHz.
  * *Expected Output:* `perch_extracted | train_emb=(1826, 3, 1536) | test_emb=(814, 3, 1536) | candidate_logits=16 | elapsed ~35s`.

* **Cell 5: Batch Extraction Official BirdNET v2.4 (Logit 4-d Summary)**
  * *Operasi:* Inisialisasi pustaka resmi `birdnet` v1.3.0 LiteRT backend. Ekstraksi logit 3-window 48 kHz dengan checkpointing chunk (`CHUNK_FILES=64`) menghasilkan matriks ringkasan 4-d: `[Bondol, Kutilang, MaxCendra (17 genus), MaxNonTarget]`.
  * *Input:* Pre-cached audio 48 kHz.
  * *Expected Output:* `birdnet_extracted | train_sum=(1826, 3, 4) | test_sum=(814, 3, 4) | elapsed ~9m | nan=0`.

* **Cell 6: Explicit Model Deletion, VRAM Purge, & Memory Sanity Check**
  * *Operasi:* Hapus variabel model Perch, BirdNET, serta tensor audio mentah (`train_32k`, `train_48k`, dll.) dari RAM setelah fitur `.npy` tersimpan. Jalankan `tf.keras.backend.clear_session()` dan `purge_vram()`. Verifikasi RAM dan VRAM siap untuk PyTorch.
  * *Input:* System memory state.
  * *Expected Output:* `memory_sanitized | audio_buffers_freed=True | ram_used < 5.0GB | pytorch_cuda_ready=True`.

---

### Tahap 3: Supervised Hybrid Baseline & In-Fold Knee-Point Thresholding (Sel 7 - 9)

* **Cell 7: In-Fold AUROC Taxa Verification (1543-d Hybrid Input) & Train Supervised Head**
  * *Operasi:* 
    1. Evaluasi kandidat taksa Cendrawasih murni *in-fold* via AUROC validasi (mengunci indeks eBird resmi `2893` dan menyusun ringkasan logit terverifikasi `7-d`).
    2. Gabungkan L2-normalized embedding `1536-d` dengan z-scored logit summary `7-d` menjadi tensor hybrid `(N, 3, 1543)`.
    3. Latih `PerchBottleneckCosineHead` (`Linear(1543, 256) -> LayerNorm -> SiLU -> Dropout(0.2) -> CosineLinear(256, 3)`) selama 40 epoch per fold dengan MIL Softmax Bag Pooling. Evaluasi pada epoch 40 (zero peeking).
  * *Input:* `train_emb (1536-d)`, `train_cand_logits (16-d)`, `metadata_with_folds.csv`.
  * *Expected Output:* `supervised_head_trained | input_dim=1543 | Global OOF Macro F1 >= 0.9283 | Bondol > 0.98 | Kutilang > 0.99 | Cendra > 0.80 | elapsed ~35s`.

* **Cell 8: Nested In-Fold Youden Knee-Point Threshold Selection (Cendrawasih & Trucuk)**
  * *Operasi:* Hitung titik infleksi matematis (*Maximum Youden Index* $J = \text{TPR} - \text{FPR}$) pada kurva logit BirdNET di dalam masing-masing fold latih:
    * `thr_cendra`: Ambang batas verifikasi panggilan Cendrawasih ($\approx -7.70$).
    * `thr_kutilang`: Ambang batas detektor Kutilang mati untuk isolasi sinyal asing Trucuk ($\approx -8.11$).
  * *Input:* `p_supervised_oof`, `train_birdnet_sum`.
  * *Expected Output:* `nested_knee_points_locked | thr_cendra_mean ~ -7.70 | thr_kutilang_mean ~ -8.11 | optimism_bias=0`.

* **Cell 9: Supervised Baseline OOF Evaluation & Anchor Test Probability Matrix**
  * *Operasi:* Evaluasi peningkatan Global OOF Macro F1 setelah *soft reassignment* Cendrawasih. Hitung matriks probabilitas inferensi awal pada 814 data uji (`p_supervised_test`) yang akan menjadi fondasi utama `sub_014` (*Safe Anchor*).
  * *Input:* Model 4 fold dari Cell 7, threshold dari Cell 8, `test_hybrid_1543`.
  * *Expected Output:* `reassigned_oof_macro_f1 >= 0.9480 | p_supervised_test shape=(814, 3)`.

---

### Tahap 4: Day 4 Core Innovation - Transductive In-Fold Self-Training (Sel 10 - 13)

* **Cell 10: Helmholtz Free-Energy Scoring E(x), kNN-5 Distance, & Consensus Inlierness Curation**
  * *Operasi:* 
    1. Hitung skor energi bebas Helmholtz $E(x) = -T \log \sum_{c=0}^2 \exp(z_c / T)$ dan jarak manifold *in-fold* kNN-5 mean-centered (`test_d_perch`) menggunakan seluruh sampel latih per fold ($N \approx 1.369$).
    2. Seleksi otomatis $\approx 120\text{--}160$ sampel test paling murni dengan **3 Lapis Filter Konsensus**: (a) Top 25% confidence tertutup, (b) Top 40% jarak manifold terdekat, dan (c) verifikasi logit BirdNET aktif di atas ambang *in-fold*.
  * *Input:* `p_supervised_test`, `train_emb`, `test_emb`, `test_birdnet_sum`.
  * *Expected Output:* `curated_pseudo_samples: N in [120, 160] | class_distribution_logged=True | zero_unknown_overlap=True`.

* **Cell 11: In-Fold Augmented Dataset Construction (Train 1543-d + Pseudo-Test per Fold)**
  * *Operasi:* Tambahkan sampel uji terkurasi beserta *hard pseudo-label*-nya ke dalam partisi `train` di masing-masing dari 4 fold, sementara partisi `val` tetap 100% murni *in-distribution*. Hitung bobot *inverse-frequency class weights* baru di setiap fold.
  * *Input:* `train_hybrid_1543`, `test_hybrid_1543[pseudo_idx]`.
  * *Expected Output:* `in_fold_augmented_ready | fold_train_sizes ~1369 + N_pseudo | fold_val_sizes ~457 (100% pure ID)`.

* **Cell 12: Retraining 4-Fold Bottleneck Cosine Heads (Self-Training 40 Epochs on 1543-d)**
  * *Operasi:* Latih ulang 4 model `PerchBottleneckCosineHead(1543 -> 256 -> 3)` dari awal (fresh initialization dengan seed terkunci) pada manifold teraugmentasi menggunakan Cosine Annealing LR selama 40 epoch per fold.
  * *Input:* Augmented datasets dari Cell 11.
  * *Expected Output:* `self_training_completed | 4 folds trained | OOF_ID_Macro_F1_maintained >= 0.9280 | elapsed < 45s`.

* **Cell 13: Self-Trained Test Inference & Adapted Probability Matrix**
  * *Operasi:* Forward pass ensemble 4 fold hasil *self-training* pada `test_hybrid_1543` menghasilkan matriks probabilitas teradaptasi domain (`p_selftrained_test`) untuk `sub_015` dan `sub_016`.
  * *Input:* 4 model self-trained dari Cell 12, `test_hybrid_1543`.
  * *Expected Output:* `p_selftrained_test shape=(814, 3) | row_sums_verified=1.0`.

---

### Tahap 5: Dual Forensic Shielding & Tri-Submission Export (Sel 14 - 17)

* **Cell 14: Dual Forensic Taxonomic Shield (Applied to Both Supervised & Self-Trained Matrices)**
  * *Operasi:* Buat fungsi modular `apply_forensic_shield(p_matrix, bnet_sum, thr_cendra, thr_kutilang)` dan terapkan secara identik pada `p_supervised_test` (untuk `sub_014`) dan `p_selftrained_test` (untuk `sub_015` & `sub_016`):
    1. *Cendrawasih Fallback:* Jika `argmax == Cendrawasih` dan logit BirdNET $< \text{thr\_cendra}$, alihkan prediksi ke kelas tertinggi kedua.
    2. *Trucuk Shield:* Jika `argmax == Kutilang` dengan confidence marjinal ($P < 0.60$) dan logit BirdNET Kutilang $< \text{thr\_kutilang}$, tandai ke dalam `trucuk_unknown_mask`.
  * *Input:* `p_supervised_test`, `p_selftrained_test`, `test_birdnet_sum`.
  * *Expected Output:* `forensic_applied_both | sup_cendra_reassigned ~85-95 | self_cendra_reassigned ~80-95 | trucuk_shielded ~70-75`.

* **Cell 15: Sweet-Spot Manifold Unknown Quantile Locking (43.5% Standard & 41.5% Probe)**
  * *Operasi:* Kalibrasi ambang batas jarak manifold gabungan $\tau$ sedemikian rupa sehingga total prediksi `Unknown` (gabungan *manifold distance* $> \tau$ dan `trucuk_unknown_mask`) mencapai:
    * Target Standar `43.5%` ($\approx 354$ sampel) untuk `sub_014` dan `sub_015`.
    * Target Konservatif `41.5%` ($\approx 338$ sampel) untuk `sub_016`.
  * *Input:* `test_d_perch`, `trucuk_unknown_mask`.
  * *Expected Output:* `quantiles_locked | count_435=354 (43.49%) | count_415=338 (41.52%)`.

* **Cell 16: Assembly, Verification, & Export Tri-Submission (sub_014, sub_015, sub_016)**
  * *Operasi:* Rakit dan ekspor 3 berkas submission resmi beserta pemeriksaan kualitas ketat (*assertions*):
    1. `experiment 14.csv` (`sub_014` - **Safe Anchor**): Pure Supervised Perch Hybrid (1543-d) + Forensic Shield @ 43.5% Unknown.
    2. `experiment 15.csv` (`sub_015` - **SOTA Champion**): In-Fold Self-Trained Perch Hybrid (1543-d) + Forensic Shield @ 43.5% Unknown.
    3. `experiment 16.csv` (`sub_016` - **Conservative Probe**): In-Fold Self-Trained Perch Hybrid (1543-d) + Forensic Shield @ 41.5% Unknown.
    * Cetak tabel perbandingan distribusi ke-4 kelas (`Bondol Jawa`, `Cendrawasih`, `Kutilang`, `Unknown`) dan persentase kesamaan (*agreement rate*) antar submission.
  * *Input:* Hasil akhir Cell 14 & Cell 15.
  * *Expected Output:* `all_3_submissions_exported | 814 rows, 0 nulls, 4 valid classes | agreement_sub14_vs_sub15 > 94.0%`.

* **Cell 17: Zip Working Directory Artifacts & Display Direct Download Link**
  * *Operasi:* Simpan ringkasan metrik eksperimen ke `day4_experiment_summary.json`, kompres seluruh artefak penting (`.csv`, `.npy`, `.json`) di `/kaggle/working` menjadi `schematics_day4_lean_artifacts.zip`, dan tampilkan `FileLink` interaktif.
  * *Input:* Seluruh file artefak di `/kaggle/working`.
  * *Expected Output:* `schematics_day4_lean_artifacts.zip created | direct download link displayed`.

---

## 4. Matriks Validasi Cepat & Keamanan Eksekusi (Fail-Safe Assertions)

| Checkpoint | Yang Diverifikasi | Kriteria Lolos (Assertion) | Tindakan Otomatis / Mitigasi |
|---|---|---|---|
| **Pasca Cell 2** | Eksistensi & Kebocoran Fold | `group_leakage == 0` & `Cendra == [35,29,16,3]` | Bangkitkan otomatis via *Exact Combinatorial Partitioning* dari `train/metadata.csv`. |
| **Pasca Cell 3** | Kepadatan Audio RAM | `zero_pad_corruption == 0.0%` | Gunakan *dynamic equidistant slicing* berdasarkan durasi asli. |
| **Pasca Cell 5** | Kelengkapan BirdNET | `np.isnan(test_sum).sum() == 0` | Ulangi chunk yang gagal via checkpoint cache. |
| **Pasca Cell 6** | Sanitasi Memori RAM & GPU | `ram_used < 6.0 GB` | Hapus buffer audio mentah (`del train_32k, train_48k`) & panggil `gc.collect()`. |
| **Pasca Cell 7** | Konsistensi Dimensi Hybrid | `X_hybrid.shape[-1] == 1543` | Gabungkan `1536-d` L2-emb + `7-d` z-scored verified taxa summary. |
| **Pasca Cell 10** | Kurasi Pseudo-Labels | `100 <= len(pseudo_idx) <= 180` | Sesuaikan kuantil batas secara adaptif jika jumlah di luar rentang. |
| **Pasca Cell 16** | Integritas & Kesamaan CSV | 814 baris, 0 null, agreement `sub_014` vs `sub_015` $\ge 90\%$ | Tolak ekspor jika ada label di luar 4 kelas resmi atau terjadi *catastrophic drift*. |

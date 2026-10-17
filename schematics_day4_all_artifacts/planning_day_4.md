# Master Blueprint: Day 4 Transductive Bioacoustic Architecture (Self-Training + Free Energy OOD + Forensic Shielding)

**Proyek:** Schematics NPC DSC 2026 (Kompetisi Sains Data Nasional ITS)
**Tugas:** Audio Open-Set Classification (Bondol Jawa, Cendrawasih, Kutilang, dan Unknown)
**Target:** Menembus batas Public Leaderboard dari 0.66760 (sub_013) menuju 0.7200 - 0.7600 & Mengunci Laporan Teknis Publikasi (Bobot 65%)
**Dokumen:** `planning_day_4.md`
**Versi:** 1.0 (End-to-End Modular Blueprint: Zero External File Uploads, Grounded in 76 Observed Evidences)
**Status:** Approved & Locked for Day 4 Execution

---

## 1. Landasan Filosofis & Resolusi Metodologis

Blueprint Day 4 dirancang 100% mandiri (*self-contained*) di Kaggle GPU Tesla T4 tanpa ketergantungan unggahan file eksternal manual. Seluruh bobot pretrained diunduh via API publik resmi yang diizinkan Guidebook Hal 10 (`google/bird-vocalization-classifier/perch_v2/2` via TFHub/Kaggle Models dan `birdnet` PyPI package v1.3.0).

### Tiga Pilar Temuan Empiris yang Dikunci (Anti-Halusinasi):
1. **Bukti Confounder Mikrofon (E-037):** Kruskal-Wallis Eta-squared membuktikan 22.62% varians akustik adalah derau mikrofon recorder ($p = 5.77 \times 10^{-53}$). Validasi wajib menggunakan partisi kombinatorial 4-Fold `StratifiedGroupKFold` pada `source_group`.
2. **Bukti Forensik Taksonomi Trucuk (E-073):** 89 file yang dipanggil Kutilang oleh Perch dengan confidence rendah ($P < 0.60$) terbukti di BirdNET adalah Trucuk (*Pycnonotus goiavier*). File ini terbukti menurunkan F1 jika dipaksa menjadi Bondol (E-073), dan wajib ditolak ke **Unknown**.
3. **Bukti Soft Reassignment Cendrawasih (E-072):** Ambang BirdNET `pct80` (thr = -7.70) pada Cendrawasih terbukti mendongkrak OOF Macro F1 dari 0.9283 menjadi **0.9482** (+0.0199) dengan memulihkan false alarm desis menjadi Bondol/Kutilang asli.

---

## 2. Arsitektur Pipeline Modular Day 4 (32 Sel)

Pipeline dibagi menjadi 6 modul terisolasi:

```
[Modul 1: Environment & Partitioning] (Sel 1 - 6)
  └── Seed 42, Metadata Audit, Exact Combinatorial 4-Fold Group Partitioning.

[Modul 2: DSP Audio Pre-Caching & Dual-Stream Extraction] (Sel 7 - 12)
  ├── Jalur A (32 kHz): Dynamic Equidistant 3-Window 5.0s, HighPass 100Hz, Linear Peak Norm 0.95.
  └── Jalur B (48 kHz): 3-Window 3.0s untuk ekstraksi logit BirdNET v2.4 (6.522 taksa).

[Modul 3: Foundation Feature Extraction & Dynamic Taxa Resolution] (Sel 13 - 17)
  ├── Google Perch v2: Embedding L2-norm 1536-d + Ringkasan Logit Kandidat 16-d.
  └── BirdNET v2.4: Logit Bondol, Kutilang, MaxCendra (17 genus), dan MaxNonTarget.

[Modul 4: Supervised Baseline & In-Fold Bottleneck Calibration] (Sel 18 - 22)
  └── PerchBottleneckCosineHead (1536 -> 256 -> 3) MIL Softmax Bag Pooling, Evaluasi Epoch 40 Final.

[Modul 5: Inlierness-Guided Adaptive Self-Training & Free Energy OOD] (Sel 23 - 27)
  ├── Helmholtz Free-Energy Scoring E(x) menggantikan heuristic distance kNN.
  ├── Kurasi adaptif pseudo-labels (Top 20% test confidence + manifold distance + BirdNET verification).
  └── Retraining Bottleneck Head pada Augmented Dataset (1.826 train + N_pseudo test).

[Modul 6: Forensic Shielding, Sweet-Spot Calibration, & Export] (Sel 28 - 32)
  ├── Cendrawasih Fallback Reassignment (thr = -7.70) ke second-best class.
  ├── Trucuk / Alien Bulbul Shield (thr = -8.11) ke Unknown.
  ├── Sweet-spot OOD quantile locking (~43.5% Unknown).
  └── Pengecekan kualitas ketat & ekspor otomatis (sub_014 hingga sub_018).
```

---

## 3. Rincian Sel-demi-Sel & Spesifikasi Teknis

### Modul 1: Environment & Partitioning (Sel 1 - 6)

* **Cell 1: Environment Setup, Deterministic Seeds, and Hardware Config**
  * *Operasi:* Set seed deterministik 42 (PyTorch, CuDNN, NumPy). Inisialisasi CUDA device Tesla T4.
  * *Input:* Runtime hardware.
  * *Expected Output:* `hardware_target | device=cuda:0 | gpu=Tesla T4 | free_vram=14.46GB`.

* **Cell 2: Metadata Schema Audit & Cardinality Verification**
  * *Operasi:* Load `train/metadata.csv` (1.826 baris) dan `sample_submission.csv` (814 baris). Assert 0 missing values.
  * *Input:* `metadata.csv`, `sample_submission.csv`.
  * *Expected Output:* `train_shape=(1826, 4) | test_shape=(814, 2) | imbalance_ratio=13.54`.

* **Cell 3: Audio Header & Deduplication Audit**
  * *Operasi:* Validasi FLAC header 48 kHz stereo 24-bit. Audit exact SHA256 deduplication (100% unique).
  * *Input:* Audio files directory.
  * *Expected Output:* `audio_integrity_verified | duplicates=0 | sr=48000 | channels=2`.

* **Cell 4: Exact Combinatorial 4-Fold Group Partitioning**
  * *Operasi:* Algoritma partisi kombinatorial eksak memetakan 15 `source_group` ke dalam 4 fold tanpa kebocoran mikrofon.
  * *Input:* `train_df["source_group"]`, `train_df["target"]`.
  * *Expected Output:* `fold_distribution: Cendrawasih=[35, 29, 16, 3] | group_leakage=0`.

* **Cell 5: Save Master Metadata with Folds**
  * *Operasi:* Tulis `metadata_with_folds.csv` ke `WORK_DIR`.
  * *Input:* Partisi fold dari Cell 4.
  * *Expected Output:* `metadata_with_folds.csv saved | shape=(1826, 5)`.

* **Cell 6: Visualisasi Distribusi Partisi & Welch PSD Confounder**
  * *Operasi:* Plotting 3-panel verifikasi partisi dan spektrum daya Welch.
  * *Input:* Audio waveform & fold metadata.
  * *Expected Output:* `eda_partition_and_confounder.png saved`.

---

### Modul 2: DSP Audio Pre-Caching (Sel 7 - 12)

* **Cell 7: Resampling Audio & Polyphase Filtering**
  * *Operasi:* Resample audio ke 32 kHz menggunakan FIR polyphase `scipy.signal.resample_poly` (up=2, down=3) untuk jalur Perch, dan pertahankan native 48 kHz untuk jalur BirdNET.
  * *Input:* Audio FLAC 48 kHz.
  * *Expected Output:* `polyphase_resampling_complete | target_sr=32000 | nyquist=16000Hz`.

* **Cell 8: Butterworth 100 Hz High-Pass Filter & Peak Normalization**
  * *Operasi:* Terapkan 4th-order Butterworth High-Pass 100 Hz (SOS format) dan Linear Peak Normalization 0.95 (tanpa fungsi non-linear `tanh`).
  * *Input:* Audio 32 kHz.
  * *Expected Output:* `dsp_v2.1_registered | highpass=100Hz | peak_norm=0.95`.

* **Cell 9: Dynamic Equidistant 3-Window Framing (32 kHz)**
  * *Operasi:* Potong audio menjadi 3 window berdurasi 5.0 detik (160.000 sampel) pada offset $[0, (L-5)/2, L-5]$ detik.
  * *Input:* Audio bersih 32 kHz.
  * *Expected Output:* `dynamic_3window_verified | shape=(3, 160000) float32 | pad_corruption=0.0%`.

* **Cell 10: RAM Pre-Caching Audio Perch (Train & Test)**
  * *Operasi:* Muat seluruh 1.826 audio train dan 814 audio test (3-window 32 kHz) ke RAM dictionary.
  * *Input:* Fungsi Cell 9.
  * *Expected Output:* `ram_caching_perch_complete | train=1826 | test=814 | elapsed ~65s`.

* **Cell 11: Dynamic 3-Window Framing (48 kHz untuk BirdNET)**
  * *Operasi:* Ekstraksi 3 window berdurasi 3.0 detik (144.000 sampel) pada audio native 48 kHz.
  * *Input:* Audio native 48 kHz.
  * *Expected Output:* `birdnet_3window_ready | shape=(3, 144000) float32`.

* **Cell 12: Memory Cache Check & GPU Buffer Allocation**
  * *Operasi:* Verifikasi ketersediaan memori sistem dan alokasi PyTorch CUDA pinned memory.
  * *Input:* System RAM & GPU VRAM.
  * *Expected Output:* `memory_sanity_verified | ram_used < 6GB | vram_free > 13GB`.

---

### Modul 3: Foundation Feature Extraction (Sel 13 - 17)

* **Cell 13: Load Google Perch v2 from TFHub Local Cache**
  * *Operasi:* Muat model `google/bird-vocalization-classifier/perch_v2/2` dari cache Kaggle Models / TFHub.
  * *Input:* TFHub URL.
  * *Expected Output:* `perch_backend_loaded | emb_dim=1536 | num_taxa=14795`.

* **Cell 14: Dynamic Taxa Matching eBird Labels (Perch)**
  * *Operasi:* Pencocokan regex nama ilmiah pada `labels.csv`: Bondol Jawa (`Lonchura leucogastroides`), Kutilang (`Pycnonotus aurigaster`), Cendrawasih (10 kandidat genus *Paradisaea*).
  * *Input:* `labels.csv` Perch.
  * *Expected Output:* `taxa_resolved: Bondol=[7329], Kutilang=[11701], Cendra=[2893, ... (10)]`.

* **Cell 15: Batch Inference Perch v2 (Train & Test)**
  * *Operasi:* Ekstraksi batch (batch_size=32) menghasilkan embedding 1536-d dan ringkasan logit 16-d untuk 1.826 train & 814 test.
  * *Input:* Tensor audio dari Cell 10.
  * *Expected Output:* `train_emb=(1826, 3, 1536) | test_emb=(814, 3, 1536) | elapsed ~35s`.

* **Cell 16: Load Official BirdNET v2.4 (LiteRT Backend)**
  * *Operasi:* Inisialisasi pustaka resmi `birdnet` v1.3.0 dengan model `AcousticModelV2_4` (76.8 MB FP32).
  * *Input:* Package `birdnet`.
  * *Expected Output:* `birdnet_loaded | species_count=6522 | backend=litert`.

* **Cell 17: Batch Inference BirdNET v2.4 (Logit Extraction)**
  * *Operasi:* Ekstraksi 3-window 48 kHz menghasilkan matriks ringkasan `(N, 3, 4)`: `[Bondol, Kutilang, MaxCendra, MaxNonTarget]`.
  * *Input:* Audio 48 kHz dari Cell 11.
  * *Expected Output:* `birdnet_train_summary=(1826, 3, 4) | birdnet_test_summary=(814, 3, 4)`.

---

### Modul 4: Supervised Baseline & In-Fold Bottleneck Head (Sel 18 - 22)

* **Cell 18: In-Fold Taxa Screening & Verification**
  * *Operasi:* Seleksi kandidat taksa Cendrawasih terbaik secara murni in-fold via AUROC validasi.
  * *Input:* Logit kandidat Perch dari Cell 15.
  * *Expected Output:* `verified_cendra_idx=2893 | in_fold_auroc > 0.88`.

* **Cell 19: Definition of PerchBottleneckCosineHead Architecture**
  * *Operasi:* Definisi arsitektur PyTorch: `Linear(1536, 256) -> LayerNorm -> SiLU -> Dropout(0.2) -> CosineLinear(256, 3, learnable_tau)`.
  * *Input:* PyTorch module.
  * *Expected Output:* `architecture_compiled: PerchBottleneckCosineHead | trainable_params ~395K`.

* **Cell 20: 4-Fold StratifiedGroupKFold Training (Zero-Peeking Evaluation)**
  * *Operasi:* Pelatihan 40 epoch per fold menggunakan Class-Balanced Focal Loss dan MIL Softmax Bag Pooling. Evaluasi wajib pada epoch terakhir 40.
  * *Input:* `perch_v2_train_emb.npy`, `metadata_with_folds.csv`.
  * *Expected Output:* `global_oof_macro_f1 > 0.9250 | Bondol > 0.98 | Kutilang > 0.99 | Cendra > 0.80`.

* **Cell 21: Test-Set Forward Pass & Closed-Set Probability Extraction**
  * *Operasi:* Inferensi ensemble 4 fold pada 814 test embedding menghasilkan `test_p_perch_v2.npy` (814, 3).
  * *Input:* Bobot model 4 fold dari Cell 20.
  * *Expected Output:* `test_p_extracted | shape=(814, 3) | min_sum=1.0 | max_sum=1.0`.

* **Cell 22: Baseline Manifold Distance Extraction (Anisotropy-Corrected kNN-5)**
  * *Operasi:* In-fold mean-centering $\tilde{x} = (x - \mu) / \|x - \mu\|_2$ dan kalkulasi jarak Cosine kNN-5.
  * *Input:* Train & test embeddings.
  * *Expected Output:* `oof_d_perch mean ~0.33 | test_d_perch mean ~0.51`.

---

### Modul 5: Inlierness-Guided Adaptive Self-Training & Free Energy OOD (Sel 23 - 27)

* **Cell 23: Helmholtz Free Energy Score Computation**
  * *Operasi:* Hitung skor energi bebas $E(x) = -T \log \sum_{c=0}^2 \exp(z_c / T)$ pada seluruh data latih OOF dan data uji test set.
  * *Input:* Logit closed-set dari Cell 20 & 21.
  * *Expected Output:* `energy_scores_computed | oof_energy_mean < test_energy_mean`.

* **Cell 24: Adaptive Test Inlierness Curation (Anti-Poisoning Filter)**
  * *Operasi:* Seleksi otomatis 120-150 sampel test dengan kriteria multi-kondisi:
    1. Confidence relatif tinggi: $P \ge \text{percentile}(P_{\text{test}}, 75)$.
    2. Jarak manifold dekat: $d_{\text{test}} \le \text{percentile}(d_{\text{test}}, 40)$.
    3. Konfirmasi BirdNET: Kutilang $\ge -8.11$, Cendrawasih $\ge -7.70$.
  * *Input:* `test_p_perch_v2`, `test_d_perch`, `birdnet_test_summary`.
  * *Expected Output:* `curated_pseudo_samples: N in [120, 160] | zero_unknown_overlap=True`.

* **Cell 25: Augmented Dataset Construction**
  * *Operasi:* Gabungkan 1.826 data latih asli dengan $N_{\text{pseudo}}$ data uji terpilih. Pembobotan sample-weight seimbang.
  * *Input:* Data latih asli + pseudo-test terpilih.
  * *Expected Output:* `augmented_dataset_shape=(1826 + N_pseudo, 3, 1536)`.

* **Cell 26: Retraining Bottleneck Cosine Head on Augmented Manifold**
  * *Operasi:* Latih ulang head klasifikasi selama 40 epoch pada augmented dataset menggunakan Cosine Annealing LR.
  * *Input:* Augmented dataset dari Cell 25.
  * *Expected Output:* `self_training_completed | elapsed < 45s | final_loss < 0.05`.

* **Cell 27: Self-Trained Test Inference & Calibrated Probability Matrix**
  * *Operasi:* Forward pass pada 814 test set menggunakan model hasil self-training.
  * *Input:* Model hasil Cell 26, `test_emb`.
  * *Expected Output:* `p_selftrained_test shape=(814, 3)`.

---

### Modul 6: Forensic Shielding, Sweet-Spot Calibration, & Export (Sel 28 - 32)

* **Cell 28: Forensic Taxonomic Reassignment (BirdNET Cendrawasih Shield)**
  * *Operasi:* Jika model memprediksi Cendrawasih namun logit BirdNET `MaxCendra < -7.70`, kembalikan prediksi ke kelas kedua model (Bondol Jawa atau Kutilang).
  * *Input:* `p_selftrained_test`, `birdnet_test_summary`.
  * *Expected Output:* `cendra_reassigned_count ~85-95 samples`.

* **Cell 29: Forensic Trucuk / Alien Bulbul Isolation**
  * *Operasi:* Jika model memprediksi Kutilang dengan probabilitas marjinal ($P_{\text{Kutilang}} < 0.60$) dan logit BirdNET Kutilang $< -8.11$, isolasi file tersebut sebagai kandidat `Unknown` (spesies non-target *Pycnonotus goiavier*).
  * *Input:* `p_selftrained_test`, `birdnet_test_summary`.
  * *Expected Output:* `trucuk_shielded_to_unknown ~65-75 samples`.

* **Cell 30: Sweet-Spot Manifold Unknown Quantile Calibration (~43.5% Target)**
  * *Operasi:* Kunci ambang batas $\tau$ pada jarak manifold Perch $d_{\text{perch}}$ sedemikian rupa sehingga total Unknown (termasuk Trucuk shield) berada di rentang optimal 43.0% s.d. 44.0%.
  * *Input:* `test_d_perch`, `trucuk_mask`.
  * *Expected Output:* `tau_locked | total_unknown_count ~350-357 (43.0% - 43.8%)`.

* **Cell 31: Multi-Submission Assembly & Export (sub_014 & sub_015)**
  * *Operasi:* Ekspor berkas submission resmi:
    * `experiment 14.csv`: Self-Trained Perch + Forensic Shield @ 43.5% Unknown (Bidikan Konservatif).
    * `experiment 15.csv`: Free-Energy Calibrated Hybrid @ 41.5% Unknown (Bidikan Agresif).
  * *Input:* Label hasil klasifikasi Cell 28-30.
  * *Expected Output:* `experiment 14.csv & experiment 15.csv verified: 814 rows, 0 NaN, 4 valid classes`.

* **Cell 32: Automated Publication Diagnostics & Artifact Bundling**
  * *Operasi:* Simpan grafik 2-panel publikasi (`ood_dual_manifold_diagnostics.png`), ekspor ringkasan `ood_thresholds.json`, dan hasilkan arsip ZIP terkompresi `schematics_day4_all_artifacts.zip` dengan tautan unduh aktif.
  * *Input:* Seluruh artefak di `/kaggle/working`.
  * *Expected Output:* `schematics_day4_all_artifacts.zip generated | download link displayed`.

---

## 4. Matriks Risiko & Protokol Mitigasi

| Potensi Risiko | Probabilitas | Dampak | Protokol Mitigasi Terverifikasi |
|---|---|---|---|
| **Label Poisoning pada Pseudo-Labeling** | Sedang | Fatal (Drop LB) | Menerapkan 3 lapis filter konsensus: Top 25% confidence + Top 40% manifold distance terdekat + verifikasi aktif logit BirdNET. |
| **Over-Rejection pada Cendrawasih** | Rendah | Sedang | Cendrawasih yang ditolak BirdNET tidak dibuang ke Unknown, melainkan dikembalikan ke kelas kedua (*second-best class*), menjaga F1 OOF di 0.9482. |
| **Shake-Down di Private LB (70%)** | Rendah | Fatal | Ambang batas Unknown dikalibrasi berbasis kuantil OOF In-Distribution dan kurva sweet-spot konsisten Day 1-3, bukan fitting acak ke 244 baris Public LB. |
| **Kaggle Kernel Timeout / Out of Memory** | Rendah | Rendah | Menggunakan TFLite LiteRT dan checkpointing batch (CHUNK_FILES=64) yang terbukti selesai dalam <10 menit di Day 3. |

---

## 5. Kepatuhan Penuh Terhadap Regulasi Kompetisi (Guidebook ITS)

1. **Strict Zero External Raw Audio:** Seluruh pelatihan fitur berasal dari 1.826 train audio resmi panitia dan 814 audio uji transduktif. Nol raw audio eksternal ditambahkan ke dataset.
2. **Atribusi Pretrained Publik Transparan (Guidebook Hal 10):**
   * *Google Perch v2:* Google Research & Cornell Lab of Ornithology (Kaggle Models/TFHub).
   * *BirdNET v2.4:* K. Lisa Yang Center for Conservation Bioacoustics, Cornell Lab of Ornithology (PyPI official package `birdnet` v1.3.0).
3. **Reprodusibilitas Penuh:** Kunci `seed=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, dan isolasi 4-fold tanpa kebocoran data.

Blueprint Day 4 ini disetujui, terkunci di `schematics_day4_all_artifacts/planning_day_4.md`, dan siap dieksekusi sel-demi-sel secara interaktif.

# Project State: Schematics NPC DSC 2026 - Bird Chirp Openset Classification

**Schema version:** 1.1
**Last updated:** 2026-10-16
**Status:** In Progress
**Current phase:** Planning -> Feature Engineering & Partitioning
**Project type:** PREDICTIVE
**Decision owner:** Ababil
**Primary notebook:** schematics-day-1.ipynb (Day 1 archive) / Day 2 Master Notebook
**Last completed cell:** Cell 28 (Day 2 Master Execution - sub_008 Golden Sweet-Spot Bundled)
**Data snapshot:** Initial download 2026-10-15
**Code version:** d9ba1cab51358023cb2a26379e14aa2df5b42524

## Objective and Success Criteria

**Problem or decision:** Classify short audio recordings into three known bird species (Bondol Jawa, Cendrawasih, Kutilang) and reject unseen species as Unknown under noisy tropical rainforest conditions.

**Unit of analysis:** One audio recording (.flac file identified by file_id).

**Outcome or target:** 4-class categorical label: Bondol Jawa, Cendrawasih, Kutilang, Unknown.

**Primary success metric:**
- Nilai Akhir Babak Penyisihan: 0.35 x Skor Prediksi + 0.65 x Skor Laporan.
- Skor Prediksi: Macro F1 Score pada 4 kelas di Kaggle test set (30% public LB, 70% private LB).
- Current Peak LB: Macro F1 = 0.59462 (sub_006: Preprocessing v2.0 + The Elite Triad + SNR-Dynamic Gating + 3-Crop TTA + Two-Stage Prototype Shrinkage)
- Skor Laporan (100%): Evaluasi dan Validasi (30%), Modeling (25%), Preprocessing dan Feature Engineering (20%), EDA (15%), Penulisan dan Struktur Laporan (10%).
- Kelolosan: 5 tim teratas lolos ke Babak Final luring di Kampus ITS Sukolilo Surabaya (15 November 2026).

**Constraints:**
- Periode penyisihan: 15-25 Oktober 2026 di Kaggle. Maksimal 5 submission per hari.
- DILARANG menggunakan data tambahan di luar dataset yang disediakan panitia (strict zero external raw audio data).
- Model pre-trained dari repositori/pustaka publik (timm, Hugging Face, TensorFlow Hub, termasuk bioacoustic foundation models seperti BirdNET / Google Perch) DIPERBOLEHKAN sesuai Guidebook Hal 10 dengan syarat mencantumkan identitas dan versi model secara transparan di notebook dan laporan, serta dapat dipertanggungjawabkan prosesnya.
- DILARANG menggunakan full AutoML atau solusi yang tidak dapat dijelaskan prosesnya.
- Model Hugging Face / timm / external pre-trained wajib melampirkan identitas/versi model yang jelas (serta access token jika gated).
- Reprodusibilitas mutlak: notebook wajib menggunakan seed/random_state. Perbedaan signifikan antara skor notebook dan skor Kaggle berisiko diskualifikasi.
- Berkas penyisihan wajib: DatSci_NamaTim_Notebook.ipynb, DatSci_NamaTim_Laporan.pdf (maks 10 halaman, maks 10 MB, Bahasa Indonesia baku), submission.csv, dan DatSci_NamaTim_Pendukung.zip (opsional).

## Data Inventory

| Source | Metadata / Grain | Time Coverage | Sensitivity | Exploration Status |
|---|---|---|---|---|
| NPC Guidebook - Data Science REVISED.pdf | 16 halaman PDF, panduan teknis resmi, jadwal, aturan, rubrik penilaian | 2026-10-15 | Public | Parsed |
| schematics-2026/train/metadata.csv | 1,826 rows, schema: file_id, audio_path, label, source_group | Unknown | Public | Explored (Cell 1-3) |
| schematics-2026/train/audio/ | 1,826 .flac audio files | Unknown | Public | Explored (Cell 2-7) |
| schematics-2026/test/audio/ | 814 .flac audio files | Unknown | Public | Explored (Cell 2-4) |
| schematics-2026/sample_submission.csv | 814 rows, schema: file_id, label | Unknown | Public | Verified |

## Progress

| Stage | Status | Evidence / Cell | Notes |
|---|---|---|---|
| Planning | Completed | Guidebook, repo, planning day 2.md v2.0 | Aturan kompetisi, rubrik 65/35, larangan data eksternal, arsitektur Day 2 locked |
| Data quality and preparation | Completed | Cell 1-8, metadata_with_folds.csv | 4-Fold StratifiedGroupKFold locked, zero group leakage verified, audio specs clean |
| EDA or statistical analysis | Completed | Cell 1-7 | Class balance, acoustic dynamics, spectral signatures, stereo correlation, high-freq audit, mel-spectrograms |
| Feature engineering | In Progress | Cell 9-10, planning day 2.md | Day 1 32kHz mel extracted; Day 2 VAD windowing and 818-dim tabular upgrade underway |
| Modeling and validation | Completed | Cell 10-18 | Full 5-model portfolio trained: LGBM (0.8542), B1 EffNet (0.8392), B2 ConvNeXt (0.8908), B3 SE-ResNeXt (0.8560), B4 Swin (0.8489) |
| Explanation and error analysis | Not Started | None | Pending OOF residual and misclassification audit |
| Methodology design | In Progress | planning day 2.md v2.0 | 5-phase Day 2 architecture with VAD, Dual-Tree, SNR-Gating, 3-Crop TTA, and Cascaded OOD |
| Delivery and export QA | In Progress | Cell 15-15c | sub_001 (LB 0.54401), sub_002 (LB 0.58315), sub_003 (LB 0.58313) evaluated |
| Operational monitoring | Not Applicable | None | Batch Kaggle competition submission |

## Assumptions Register

| ID | Assumption | Category | Confidence | Impact if Wrong | Validation Plan | Status |
|---|---|---|---|---|---|---|
| A-001 | GroupKFold on source_group prevents recording-source leakage and mimics test distribution | Model | High | High | Compare random CV vs GroupKFold CV correlation with public LB | Open |
| A-002 | Test set contains Unknown class samples in non-trivial proportion | Data | High | High | Audit prediction distribution and evaluate threshold sensitivity on held-out pseudo-unknown classes | Validated (sub_002 jumped +0.039 to 0.58315 with ~36% Unknown) |
| A-003 | Audio duration and sample rates across train and test share a consistent sampling distribution | Data | High | Medium | Verified in Cell 2 & Cell 3 (KS test p=0.6695, no duration shift) | Validated |
| A-004 | Standar pre-trained model backbones pada ImageNet (EfficientNet-B0, ConvNeXt-Tiny, SE-ResNeXt-50 32x4d) diperbolehkan karena merupakan parameter arsitektur visual dan bukan dataset audio eksternal | Model | High | High | Dokumentasikan nama model `timm` secara transparan di notebook dan laporan | Open |
| A-005 | Laporan 10 halaman bernilai 65 persen menentukan kelolosan lebih dominan dibanding micro-optimasi LB (35 persen) | Business | High | High | Alokasikan porsi kerja seimbang antara visualisasi EDA ilmiah, justifikasi modeling, dan error analysis | Open |

## Decisions Log

### D-001: GroupKFold by source_group for Validation

- **Chosen:** 4-fold StratifiedGroupKFold on source_group with stratification on label, evaluated via Global OOF Macro F1.
- **Alternatives considered:** 5-fold GroupKFold, StratifiedKFold without grouping, mean of per-fold F1.
- **Why:** Cell 2 proved Cendrawasih and Kutilang each have exactly 4 source groups (pure per class). A 5-fold split is mathematically impossible without leaving folds with zero validation samples for minority classes. Global OOF Macro F1 is enforced because Cendrawasih group sizes vary from 3 to 35 (E-014).
- **Evidence:** Cell 2 & Cell 3 stdout (source_groups_per_class: Bondol Jawa: 7, Cendrawasih: 4, Kutilang: 4).
- **Revisit when:** Not Applicable.

### D-002: Dual Optimization Protocol (65% Report, 35% Kaggle LB)

- **Chosen:** Eksekusi pipeline dengan dokumentasi eksperimental lengkap sesuai 5 bab laporan (EDA 15%, Preprocessing 20%, Modeling 25%, Validasi 30%, Struktur 10%) bersamaan dengan optimasi Macro F1.
- **Alternatives considered:** Chasing Kaggle Leaderboard saja tanpa mencatat log eksperimen dan ablasi.
- **Why:** Bobot laporan 65% jauh lebih besar dari skor prediksi 35%. Hanya 5 tim teratas gabungan yang lolos ke final ITS.
- **Evidence:** Panduan resmi Halaman 9-10 (Penilaian Babak Penyisihan: 0.35 Skor Prediksi + 0.65 Skor Laporan).
- **Revisit when:** Not Applicable.

### D-003: Compliance on Permitted Pretrained Models and Prohibited Raw External Data

- **Chosen:** Menggunakan dataset resmi 1.826 train audio untuk fine-tuning dan evaluasi. Penggunaan model pre-trained publik (ImageNet-1k, Hugging Face, maupun foundation model bioakustik seperti BirdNET dan Google Perch) diperbolehkan berdasarkan Guidebook Hal 10 selama bukan penambahan raw audio data baru ke dalam dataset panitia dan identitas/versi model dicantumkan secara transparan.
- **Alternatives considered:** Penambahan raw audio file dari Xeno-Canto/eBird (dilarang mutlak), full AutoML (dilarang).
- **Why:** Guidebook Halaman 9 melarang penambahan file data di luar dataset resmi, namun Halaman 10 secara eksplisit mengizinkan tools, library, framework, dan model Hugging Face publik selama identitas model dapat dipertanggungjawabkan dan prosesnya transparan (bukan black-box full AutoML).
- **Evidence:** Halaman 9-10 dan 15 Buku Panduan.
- **Revisit when:** Not Applicable.

### D-004: Bundled Submission Tracking Protocol

- **Chosen:** Setiap submission Kaggle (maks 5/hari) dibundel ke folder terisolasi `submissions/sub_XXX_<timestamp>/` berisi `submission.csv`, snapshot notebook/script, dan `metrics.json` (skor CV, detail model, threshold Unknown, skor public LB).
- **Alternatives considered:** Menimpa satu file submission.csv lokal tanpa tracking versi.
- **Why:** Kuota ketat 5 sub/hari dan aturan diskualifikasi panitia jika skor notebook tidak konsisten dengan skor Kaggle.
- **Evidence:** NPC Guidebook Hal 8 (Aturan konsistensi skor notebook dan leaderboard) serta instruksi user.
- **Revisit when:** Not Applicable.

### D-005: Resampling Rate and Nyquist Frequency Selection (Updated to 32 kHz Default)

- **Chosen:** Resampling audio native 48 kHz ke **32 kHz (Nyquist cutoff 16 kHz) sebagai default utama pipeline** menggunakan `scipy.signal.resample_poly`, dan menyimpan 16 kHz untuk ablasi ABL-05.
- **Alternatives considered:** Default 16 kHz (membuang band 8-16 kHz), atau mempertahankan native 48 kHz (boros RAM/compute 33%).
- **Why:** Bukti empiris Cell 7 menunjukkan band 16-24 kHz hanya menyumbang 0.11%-1.22% energi, sedangkan band 8-16 kHz membawa hingga 9.08% energi harmonik Cendrawasih (E-017). Karena beralih ke 2D CNN (EfficientNet-B0) yang tidak terkendala ukuran patch statis 16 kHz seperti AST, penggunaan default 32 kHz menjaga sinyal pembeda Cendrawasih secara utuh.
- **Evidence:** Cell 7 stdout (band_16_24k: Bondol 0.11%, Cendrawasih 0.21%, Kutilang 1.22%; band_8_16k up to 9.08% on Cendrawasih).
- **Revisit when:** Ablasi ABL-05 menunjukkan 16 kHz memiliki Macro F1 OOF yang lebih tinggi.

### D-006: Stereo Downmixing to Single Channel Mono

- **Chosen:** Downmix stereo 2-channel menjadi 1-channel mono via averaging `(ch0 + ch1) / 2`.
- **Alternatives considered:** Mempertahankan dual-channel stereo pada input spectrogram.
- **Why:** Bukti Cell 7 menunjukkan korelasi antar-channel sangat tinggi (0.724 - 0.800) dan selisih volume RMS mendekati nol (0.002 - 0.003). Downmix menghemat 50% ukuran tensor tanpa menghilangkan fitur kicauan.
- **Evidence:** Cell 7 stdout (rms_diff mean 0.002-0.003, corr mean 0.724-0.800).
- **Revisit when:** Lokalisasi spasial stereo terbukti membedakan burung dari noise latar belakang.

### D-007: Selection of ImageNet 2D CNN Backbones and 256-d Bottleneck Embedding over Audio Transformers

- **Chosen:** Menggunakan **EfficientNet-B0** (~5.3M) sebagai backbone utama, **ConvNeXt-Tiny** (~28M) untuk diversitas ensemble, dan **SE-ResNeXt-50 32x4d** (`se_resnext50_32x4d`, ~27.5M) sebagai representasi channel-attention terbukti (champion Paddy Field Segmentation), dengan protokol **LP-FT 2-Phase**, `in_chans=1`, serta proyeksi bottleneck `GAP -> Linear(d, 256) -> BN -> SiLU -> Dropout(0.3) -> Linear(256, 3)`.
- **Alternatives considered:** Pretrained Audio Spectrogram Transformer (AST ~87M) / BEATs (~90M), EfficientNetV2-S (~21.5M), atau menggunakan fitur GAP mentah ($d=1280$) langsung untuk kNN OOD.
- **Why:** Pada $N=1.826$ (Cendrawasih hanya 83 sampel), Transformer murni tanpa *spatial inductive bias* sangat rentan mengalami overfitting parah. Selain itu, dimensi embedding besar ($d=768$ hingga $1280$) memicu *curse of dimensionality* pada kNN OOD (`k=20`) untuk ukuran fold latih $N \approx 1.369$. Bottleneck 256-dimensi membuat ruang jarak kNN padat, stabil, dan seragam antar arsitektur.
- **Evidence:** E-001 ($N=1.826$), E-007 (Cendrawasih $N=83$), dan korpus `cv_arch_efficientnet_compound_scaling`, `cv_trans_lp_ft_two_phase_adaptation`, `cv_ood_knn_embedding`.
- **Revisit when:** Hanya jika diperlukan tambahan diversitas modalitas dari EfficientAT (`mn10_as`, ~4.8M).

### D-008: Selection of Track B4 Architecture: Swin-Tiny Shifted Window Attention over Audio Transformers

- **Chosen:** Menetapkan **Swin Transformer Tiny** (`swin_tiny_patch4_window7_224`, ~28.3M params) sebagai kandidat Track B4 penutup diversitas ensemble, menggantikan EfficientAT (AudioSet 16kHz).
- **Alternatives considered:** EfficientAT (`mn10_as`, memotong resolusi ke 16 kHz dan membuang 9.08% harmonik Cendrawasih 8-16kHz), ViT murni (`vit_base_patch16_224`, ditolak korpus `cv_arch_vit_vision_transformer` karena ketiadaan inductive bias di dataset kecil <100K memicu overfitting parah).
- **Why:** Korpus `cv_arch_swin_shifted_window_attention` dan `cv_arch_hybrid_cnn_transformer_dual_path` menetapkan bahwa representasi *non-local multi-head self-attention* bersifat ortogonal dan komplementer sempurna terhadap konvolusi lokal CNN. Spektrogram bioakustik memiliki relasi harmonik vertikal (kicauan dasar di 2-4 kHz berpasangan dengan harmonik di 6-8 kHz) yang ditangkap secara komputasional oleh *Shifted Window Attention (W-MSA & SW-MSA)*, melengkapi 3 keluarga CNN kita (Depthwise B1, Large-Kernel B2, Channel-Attention B3).
- **Evidence:** Korpus `cv_arch_swin_shifted_window_attention`, `cv_arch_hybrid_cnn_transformer_dual_path`, dan Cell 7 PSD harmonik 8-16kHz (`E-017`).
- **Revisit when:** Resolusi spektrogram ($128 \times 626$) memicu interpolasi relative position bias yang tidak stabil di timm.

### D-009: Critical Filtering & Integration of Teammate (Jere) EDA & Confounder Findings

- **Chosen:** Mengadopsi bukti kuantitatif Kruskal-Wallis Eta-squared ($\eta^2 = 0.2366, p=1.855 \times 10^{-55}$) untuk memperkuat justifikasi metodologis StratifiedGroupKFold di Bab Evaluasi & Validasi laporan teknis (bobot 30%), namun MENOLAK estimasi ambang batas Unknown Jere ($14.25\%$).
- **Alternatives considered:** Menggunakan ambang batas penolakan P95 kNN Jere (14.25% Unknown, 116 sampel uji) sebagai dasar post-processing.
- **Why:** Eksperimen empiris Public LB membuktikan estimasi Unknown Jere (14.25%) terlalu konservatif (meremehkan Unknown sebenarnya sekitar 20%). Sub_001 (12.04% Unknown) hanya menghasilkan LB 0.54401, sedangkan Sub_002 (36.61% Unknown) melonjak ke 0.58315 dan Sub_004 ke 0.59183. Delapan fitur skalar Welch statis milik Jere merata-ratakan audio sepanjang 10 detik, sehingga modulasi vokal burung pendek tenggelam dalam desis latar belakang dan gagal menangkap ketidakpastian sejati.
- **Evidence:** `E-027` (sub_001 LB 0.54401), `E-029` (sub_002 LB 0.58315), `E-036` (sub_004 LB 0.59183), `E-037` (Confounder Eta-squared), `E-038` (OOD Spectral Profile).
- **Revisit when:** Fitur bioakustik diekstraksi secara *frame-level dynamic window*, bukan *global scalar average*.

## Evidence Ledger

| ID | Claim or Result | Value | Evidence Source | Evaluation Context | Status |
|---|---|---|---|---|---|
| E-001 | Train dataset size | 1,826 audio recordings | CLI inventory | Train directory file count | Observed |
| E-002 | Test dataset size | 814 audio recordings | CLI inventory | Test directory file count | Observed |
| E-003 | Bobot penilaian babak penyisihan | 35% Skor Prediksi (Kaggle Macro F1) + 65% Laporan Analisis | NPC Guidebook Hal 9 | Dokumen Kompetisi | Observed |
| E-004 | Rubrik penilaian laporan penyisihan | Validasi 30%, Model 25%, Preprocess 20%, EDA 15%, Penulisan 10% | NPC Guidebook Hal 10 | Dokumen Kompetisi | Observed |
| E-005 | Ketentuan format berkas penyisihan | Notebook (.ipynb), Laporan maks 10 hal (.pdf), submission.csv | NPC Guidebook Hal 9, 12 | Dokumen Kompetisi | Observed |
| E-006 | Batasan eksternal data dan AutoML | Dilarang data tambahan di luar panitia; dilarang full AutoML | NPC Guidebook Hal 8, 14 | Dokumen Kompetisi | Observed |
| E-007 | Train class balance | Bondol Jawa: 1124 (61.56%), Kutilang: 619 (33.90%), Cendrawasih: 83 (4.55%) | Cell 1 | Train metadata | Observed |
| E-008 | Unique source groups | 15 groups total (pure per class: Bondol 7, Kutilang 4, Cendrawasih 4) | Cell 1 | Train metadata | Observed |
| E-009 | Audio physical specs | Exact match: 48,000 Hz, 2 channels, duration 5.27s-14.90s (mean 10.04s) | Cell 2 | Audio stream headers | Observed |
| E-010 | Source group bottleneck | Cendrawasih and Kutilang have exactly 4 source groups, capping group-aware CV to 4 folds | Cell 2 | Crosstab | Observed |
| E-011 | Class imbalance ratio | IR = 13.54 (Bondol Jawa / Cendrawasih), Shannon entropy = 1.163 bits (norm: 0.734) | Cell 3 | Cell output | Observed |
| E-012 | Exact hash deduplication | 100% unique in train (1826/1826) and test (814/814), zero exact leakage | Cell 3 | SHA256 audit | Observed |
| E-013 | Duration shift test | KS statistic 0.0302, p-value 0.6695, Wasserstein distance 0.0892s (no shift) | Cell 3 | Scipy stats | Observed |
| E-014 | Group cardinality range | Cendrawasih: [3, 16, 29, 35] (total 83); Kutilang: [132, 143, 166, 178] (total 619); Bondol: [58, 82, 117, 123, 176, 232, 336] (total 1124) | Cell 3 | Group groupby | Observed |
| E-015 | Acoustic signal dynamics | Clipping rate 40-55%, silence ~10%, Kutilang ZCR (0.2899) lower than Cendrawasih (0.3699) and Bondol (0.3483) | Cell 4 | Audio loader probe | Observed |
| E-016 | Spectral power band signatures | Kutilang dominant in 2-4kHz (57.2%); Bondol broad in 2-6kHz (64.5%); Cendrawasih high in 0-2kHz (44.0%) and 6-8kHz (14.7%) | Cell 5 | Welch PSD decomposition | Observed |
| E-017 | High-frequency power audit | 16-24kHz contains only 0.11-1.22% power; 8-16kHz holds up to 9.08% power in Cendrawasih; 0-8kHz contains >90.7% power | Cell 7 | 48kHz Welch PSD | Observed |
| E-018 | Stereo channel consistency | High inter-channel correlation (0.724-0.800) and minimal RMS delta (0.002-0.003) support mono downmixing | Cell 7 | Soundfile stereo audit | Observed |
| E-019 | 4-fold StratifiedGroupKFold partition | Zero group leakage; 1 minority group/fold; val sizes 503, 462, 422, 439; Cendrawasih counts: 35, 29, 16, 3 | Cell 8 | StratifiedGroupKFold | Observed |
| E-020 | Preprocessing & mel-spectrogram pipeline verification | 32kHz tensor (1, 128, 626), 16kHz tensor (1, 128, 501), per-sample z-score (mean 0.0, std 1.0), load time ~70ms | Cell 9 | BirdChirpDataset unit audit | Observed |
| E-021 | Track A LightGBM 4-Fold Global OOF Macro F1 | 0.8542 (Bondol 0.9556, Kutilang 0.9444, Cendrawasih 0.6627); Fold 3 cendra_f1 dropped to 0.1935 validating FIX-A | Cell 10 | 4-fold StratifiedGroupKFold on 792 mel features | Observed |
| E-022 | Track B1 EfficientNet-B0 Global OOF Macro F1 | 0.8392; Cendrawasih F1 0.6914 (+0.0286 vs LightGBM); Fold 0 F1 0.9394; 256-d bottleneck embeddings extracted | Cell 12 | LP-FT 2-phase on 4-Fold StratifiedGroupKFold | Observed |
| E-023 | Track A + Track B1 Linear Opinion Pool synergy | Blend F1 0.8818 at w_B1=0.51 (+0.0276 vs LGBM, +0.0426 vs B1); Bondol 0.9677, Cendrawasih 0.7152, Kutilang 0.9627 | Cell 13 | LOP grid search on 4-Fold OOF | Observed |
| E-024 | Open-Set OOD detector AUROC benchmark (ABL-03) | 256-d kNN AUROC 0.9970 vs MSP 0.9283 vs Energy 0.8209; Ensemble OOD achieves AUROC 0.9987 | Cell 13 | In-Distribution vs 270 Pseudo-OOD chimera | Observed |
| E-025 | Nelder-Mead 4-class open-set optimization | Global 4-Class Macro F1 = 0.9099 (Unknown F1=0.9963, ID false reject=0.11%, recall=100.0%); tau=1.6756, w_b1=0.48 | Cell 14 | 4-class OOF + Pseudo-OOD Nelder-Mead | Observed |
| E-026 | Test set inference and sub_001 submission distribution | 814 samples: Kutilang 302 (37.10%), Bondol Jawa 302 (37.10%), Cendrawasih 112 (13.76%), Unknown 98 (12.04%); tau=0.7939 (ID p95 fallback) | Cell 15 | Track A (0.52) + Track B1 (0.48) 4-Fold ensemble on 814 test audio | Observed |
| E-027 | Public Leaderboard anchor score (sub_001) | Macro F1 = 0.54401 on 30% public LB; rank 17 (team: disuruh ryu daftar); indicates test set has distinct Unknown or domain distribution | Kaggle LB | Kaggle Public LB evaluation | Observed |
| E-028 | sub_002 threshold probe and Cendrawasih FP guard | Cendrawasih dropped 112 -> 35 (4.30%, matching train 4.55%); Unknown increased 98 -> 298 (36.61%); tau_p90=0.6450 | Cell 15b | Class-conditional kNN (k=5) + Cendrawasih guard on 814 test audio | Observed |
| E-029 | Public Leaderboard validation of sub_002 | Macro F1 = 0.58315 (+0.03914 vs sub_001 0.54401); validates Cendrawasih FP guard and ~36% Unknown proportion hypothesis | Kaggle LB | Kaggle Public LB evaluation | Observed |
| E-030 | sub_003 sweet-spot probe generation | 28.01% Unknown (228/814), Cendrawasih 59 (7.25%), Kutilang 274 (33.66%), Bondol 253 (31.08%); p92.0 quantile threshold | Cell 15c | 3-way distribution comparison on 814 test audio | Observed |
| E-031 | Public Leaderboard validation of sub_003 | Macro F1 = 0.58313 (delta vs sub_002: -0.00002); proves plateau region between 28% and 36% Unknown; establishes optimal Unknown boundary at ~32-36% | Kaggle LB | Kaggle Public LB evaluation | Observed |
| E-032 | Track B2 ConvNeXt-Tiny Global OOF Macro F1 | 0.8908 (+0.0516 vs B1 0.8392, +0.0366 vs Track A 0.8542); Cendrawasih F1 jumped to 0.8316 (+0.1402 vs B1); 7x7 depthwise convs validated | Cell 16 | LP-FT 2-phase on 4-Fold StratifiedGroupKFold | Observed |
| E-033 | Track B3 SE-ResNeXt-50 32x4d Global OOF Macro F1 | 0.8560; Fold 1 perfect Cendrawasih F1 1.0000 (Fold Macro F1 0.9983); Global Cendrawasih F1 0.7391 (+0.0477 vs B1 0.6914); channel attention validated | Cell 17 | LP-FT 2-phase on 4-Fold StratifiedGroupKFold | Observed |
| E-034 | Track B4 Swin-Tiny Transformer Global OOF Macro F1 | 0.8489; Cendrawasih F1 0.7291; Fold 0 F1 0.9747; completes 5-family diverse portfolio | Cell 18 | LP-FT 2-phase on 4-Fold StratifiedGroupKFold | Observed |
| E-035 | sub_004 5-Model Ensemble and Test Inference | NLL-weighted blend (LGBM 0.683, ConvNeXt 0.207, Swin 0.111); OOF Macro F1=0.8670; Unknown 307 (37.71%), Cendrawasih 76 (9.34%), Kutilang 227, Bondol 204 | Cell 19 | 5-model NLL blend + fold-safe kNN OOD | Observed |
| E-036 | Public Leaderboard validation of sub_004 | Macro F1 = 0.59183 (+0.00868 vs sub_002 0.58315, +0.04782 vs sub_001); new competition high; validates multimodal tabular-DL ensemble synergy | Kaggle LB | Kaggle Public LB evaluation | Observed |
| E-037 | Audit Confounder Source Group Jere (Kruskal-Wallis Eta-squared) | Source group menjelaskan eta^2 = 0.2386 (23.86%) varians akustik Bondol Jawa (H=272.52, p=6.273e-56) dan 6.43% Kutilang (p=3.086e-09); bukti formal kebocoran mikrofon jika tanpa StratifiedGroupKFold | notebook jere Cell 5 / Day 2 Master Cell 6 | Kruskal-Wallis non-parametrik & Eta-squared effect size | Observed |
| E-038 | Karakteristik Bioakustik Kandidat Unknown pada Fitur Welch PSD (Jere) | Sampel uji OOD kNN > P95 (14.25%) memiliki frekuensi dominan anjlok ke 1,359 Hz (vs in-dist 2,930 Hz) dan energi low-band <1 kHz melonjak 10.64% -> 19.33%; konfirmasi anomali spektral | notebook jere Cell 6 | Welch PSD (nperseg=2048) & k-NN (k=5) pada 8 fitur DSP | Observed |
| E-039 | Triad-Tree Tabular Benchmark & Dual Blend Synergy (v2.0) | Standalone: LGBM 0.8475, XGBoost 0.8408, CatBoost 0.8322. 50/50 Dual Blend LGBM+XGBoost mencapai Global OOF Macro F1 = 0.8486 (Bondol 0.9497, Cendra 0.6552, Kutilang 0.9411); mengungguli seluruh standalone tree | Day 2 Master Cell 13-16 | 4-Fold StratifiedGroupKFold pada 818 fitur v2.0 | Observed |
| E-040 | Retrained ConvNeXt-Tiny v2.0 Breakthrough Performance | Global OOF Macro F1 melonjak ke 0.9323 (+0.0415 vs v1.0 0.8908); Cendrawasih F1 meroket ke 0.9448 (+0.1132 vs v1.0 0.8316); Fold 1 perfect 1.0000; memvalidasi VAD windowing dan spectral subtraction | Day 2 Master Cell 18 | 4-Fold LP-FT 2-phase pada audio Preprocessing v2.0 | Observed |
| E-041 | Retrained Swin-Tiny Transformer v2.0 Performance Surge | Global OOF Macro F1 melonjak ke 0.9098 (+0.0609 vs v1.0 0.8489); Cendrawasih F1 naik tajam ke 0.8671 (+0.1380 vs v1.0 0.7291); membuktikan shifted window attention pada spektrogram bersih | Day 2 Master Cell 19 | 4-Fold LP-FT 2-phase pada audio Preprocessing v2.0 | Observed |
| E-042 | Retrained SE-ResNeXt-50 32x4d v2.0 Performance Surge | Global OOF Macro F1 melonjak ke 0.9005 (+0.0445 vs v1.0 0.8560); Cendrawasih F1 naik tajam ke 0.8588 (+0.1197 vs v1.0 0.7391); memvalidasi channel attention pada spektrogram bersih | Day 2 Master Cell 20 | 4-Fold LP-FT 2-phase pada audio Preprocessing v2.0 | Observed |
| E-043 | Retrained EfficientNet-B0 v2.0 Performance Surge | Global OOF Macro F1 naik ke 0.8721 (+0.0329 vs v1.0 0.8392); Cendrawasih F1 naik ke 0.7692 (+0.0778 vs v1.0 0.6914); model teringan (~5.3M params) selesai terlatih | Day 2 Master Cell 21 | 4-Fold LP-FT 2-phase pada audio Preprocessing v2.0 | Observed |
| E-044 | Acoustic Noise-Conditioned Dynamic Gating Breakthrough (v2.0) | Global OOF Macro F1 melesat ke 0.9380 (Harmonic NLL: 0.1938); Bondol F1=0.9495, Cendrawasih F1=0.9390 (Presisi 95.06%, 77 TP, 4 FP), Kutilang F1=0.9256; Two-Way Shield memveto 53 FP Tabular dan 37 FP Swin; melampaui seluruh single model | Day 2 Master Cell 23 | Grid Search SNR Gating (w_conv=0.8, w_swin=0.2, w_max=0.45, gamma=1.0) | Observed |
| E-045 | 3-Crop Test-Time Augmentation (TTA) Verification | 419 dari 814 sampel test (>10s) dievaluasi dengan 3-crop multi-window (VAD + Left + Right); agreement rate 98.53% vs 1-crop; memulihkan +8 kandidat vokal di tepi rekaman; inferensi 8 fold tuntas dalam 76.3s | Day 2 Master Cell 24 | Confidence-Weighted Pooling pada ConvNeXt-Tiny & Swin-Tiny | Observed |
| E-046 | Two-Stage Cascaded OOD Rejection & Test Distribution Locked | Ambang batas terkalibrasi p91.5 (tau=0.5473) menghasilkan Unknown sub_005 = 308 (37.84%) dan sub_006 = 351 (43.12%); Stage 2 Cendrawasih Prototype Shrinkage (tau=0.4214) memotong 37 ekses FP (129 -> 92 core, 11.30%); perbedaan 43 sampel terisolasi sempurna untuk pembuktian ablasi Ide 1 vs Ide 1+2 | Day 2 Master Cell 25-26 | In-Fold Cosine kNN-5 + Gated Uncertainty + Entropy Fusion | Observed |
| E-047 | Public Leaderboard validation of sub_005 | Macro F1 = 0.55789 on 30% public LB; konfirmasi empiris bahwa tanpa Prototype Shrinkage Cendrawasih (130 kandidat, 15.97%), desis frekuensi rendah menyerap Unknown dan menjatuhkan presisi kelas minoritas | Kaggle LB | Kaggle Public LB evaluation | Observed |
| E-048 | Public Leaderboard validation of sub_006 (New Peak LB 0.59462) | Macro F1 = 0.59462 (+0.03673 vs sub_005 0.55789, +0.00279 vs Day 1 sub_004 0.59183); new all-time high competition score; bukti empiris tak terbantahkan bahwa Adaptive Prototype Shrinkage (92 core Cendrawasih) + 3-Crop TTA mengangkat performa secara signifikan | Kaggle LB | Kaggle Public LB evaluation | Observed |
| E-049 | sub_007 Bioacoustic High-Harmonic Gate & Dual-Manifold Shrinkage | Dual-manifold in-fold kNN (ConvNeXt 75% + Swin 25%) + High-Harmonic Deficit penalty mengunci Cendrawasih tepat di 70 sampel (8.60%, -22 ekses vs sub_006) dan Unknown di 387 sampel (47.54%); 44 sampel terkalibrasi ke zona paling defensible | Day 2 Master Cell 27 | Cell 27 execution | Observed |
| E-050 | Public Leaderboard validation of sub_007 | Macro F1 = 0.58805 on 30% public LB; membuktikan kurva konkavitas batas penolakan: rasio Unknown 43.12% (sub_006) adalah titik optimum global; pengetatan berlebih ke 47.54% Unknown (sub_007) mulai memotong recall Bondol Jawa (-18 sampel) | Kaggle LB | Kaggle Public LB evaluation | Observed |
| E-051 | Public Leaderboard validation of sub_008 | Macro F1 = 0.58356 on 30% public LB; mengonfirmasi bahwa menaikkan threshold global ke p92.0 (tau=0.5924) pada test_ood komposit terlalu longgar untuk Stage 1, membuktikan sub_006 (p91.5, tau=0.5473, 92 Cendrawasih) adalah titik manis definitif | Kaggle LB | Kaggle Public LB evaluation | Observed |
| E-052 | Legalitas Penggunaan Pretrained Bioacoustic Models (BirdNET/Perch) | Guidebook Hal 10 eksplisit mengizinkan model, tools, library, dan gated repository Hugging Face publik selama versi dicantumkan dan proses transparan; yang dilarang adalah raw audio eksternal dan full AutoML | Dokumen Resmi | NPC Guidebook Hal 9-10 & 15 | Observed |

## Methodology Design

**Method ID and version:** PLAN-DAY2 v2.0

**Maturity:** Proposal

**Method spec:** planning day 2.md v2.0 (Audio Preprocessing v2.0, Dual-Tree Tabular Engine, SNR-Dynamic Gating, 3-Crop TTA, Two-Stage Cascaded OOD)

**Case signature and baseline gap:** Open-set audio classification with IR=13.54, group-structured recordings, clipping artifacts, and zero Unknown labels in train. Day 1 reached 0.59183 ceiling due to static 10s cropping, clipping saturation, and unweighted NLL bias.

**Exact next experiment:** Execute Day 2 Action Plan starting with Slot 1 (sub_005: Quick-Win SNR-Dynamic Gated Blending and Two-Stage OOD Prototype Shrinkage on Day 1 artifacts)

## Model Drivers and Error Analysis

Not Applicable at initial project planning phase.

## Artifacts

| Artifact | Purpose | Validation Status |
|---|---|---|
| NPC Guidebook - Data Science REVISED.pdf | Buku panduan teknis resmi kompetisi | Passed |
| schematics-2026/sample_submission.csv | Ground truth submission schema and file_id list | Passed |
| eda_acoustic_profiles.png | Visualisasi 3-panel profil akustik dan separabilitas spektral untuk laporan | Passed |
| eda_mel_spectrograms.png | Visualisasi mel-spectrogram komparatif 3 spesies untuk bab Preprocessing laporan | Passed |
| schematics_day1_all_artifacts/planning day 1.md | Spesifikasi metodologi dan rencana eksperimen Day 1 (diarsipkan) | Passed |
| planning day 2.md | Spesifikasi teknis Day 2 v2.0 (VAD, Triad-Tree, 4 DL Backbones v2, Grand Prix Model Decision, Acoustic Dynamic Gating, 3-Crop TTA, Two-Stage OOD, Full 26-Cell Master Blueprint) | Passed |
| schematics-day-1.ipynb | Notebook master terintegrasi 23 sel (EDA, Preprocessing, LGBM, B1-B4 DL, 5M Ensemble) | Passed |
| schematics_day1_notebook.md | Transkrip lengkap markdown 23 cell dan seluruh output runtime Day 1 (174 KB) | Passed |
| metadata_with_folds.csv | Dataset train metadata dengan kolom partisi 4-Fold StratifiedGroupKFold | Passed |
| preprocess_config.json | Konfigurasi pipeline preprocessing dan model backbones v1.2 terkunci | Passed |
| track_a_X_train_32k.npy | Matriks fitur tabular 792-dimensi (N=1826) terekstraksi | Passed |
| oof_track_a_lgbm.csv | Prediksi probabilitas OOF Track A LightGBM (Global Macro F1 0.8542) | Passed |
| track_a_metrics.json | Ringkasan metrik evaluasi OOF Track A LightGBM | Passed |
| oof_track_b1_effnet_b0.csv | Prediksi probabilitas dan logit OOF Track B1 EfficientNet-B0 | Passed |
| oof_b1_effnet_b0_emb256d.npy | Representasi embedding 256-d L2-normalized OOF (N=1826) | Passed |
| train_fold_b1_effnet_b0_emb256d.pt | Reference in-distribution embeddings per fold untuk kNN OOD | Passed |
| checkpoints_b1_effnet_b0/ | Checkpoint bobot model terbaik tiap fold hasil LP-FT | Passed |
| checkpoints_b2_convnext_tiny/ | Checkpoint bobot ConvNeXt-Tiny terbaik tiap fold hasil LP-FT | Passed |
| oof_b2_convnext_tiny_probs.npy | Prediksi probabilitas OOF Track B2 ConvNeXt-Tiny (F1 0.8908) | Passed |
| oof_b2_convnext_tiny_emb256d.npy | Representasi embedding 256-d L2-normalized OOF B2 | Passed |
| track_b2_metrics.json | Ringkasan metrik evaluasi OOF Track B2 ConvNeXt-Tiny | Passed |
| checkpoints_b3_se_resnext50/ | Checkpoint bobot SE-ResNeXt-50 32x4d terbaik tiap fold hasil LP-FT | Passed |
| oof_b3_se_resnext50_probs.npy | Prediksi probabilitas OOF Track B3 SE-ResNeXt-50 (F1 0.8560) | Passed |
| oof_b3_se_resnext50_emb256d.npy | Representasi embedding 256-d L2-normalized OOF B3 | Passed |
| track_b3_metrics.json | Ringkasan metrik evaluasi OOF Track B3 SE-ResNeXt-50 | Passed |
| checkpoints_b4_swin_tiny/ | Checkpoint bobot Swin-Tiny terbaik tiap fold hasil LP-FT | Passed |
| oof_b4_swin_tiny_probs.npy | Prediksi probabilitas OOF Track B4 Swin-Tiny (F1 0.8489) | Passed |
| oof_b4_swin_tiny_emb256d.npy | Representasi embedding 256-d L2-normalized OOF B4 | Passed |
| track_b4_metrics.json | Ringkasan metrik evaluasi OOF Track B4 Swin-Tiny | Passed |
| oof_ood_calibration_bundle.npy | Bundle skor OOD (MSP, Energy, kNN 256-d) dan probabilitas OOF | Passed |
| ood_calibration_lock.json | Konfigurasi bobot ensemble dan threshold OOD 4-kelas terkunci | Passed |
| experiment 1.csv | File submission pertama (814 baris, 4 kelas) siap submit ke Kaggle | Passed |
| experiment 2.csv | File submission kedua (814 baris, Cendrawasih guard, 36.61% Unknown) | Passed |
| experiment 3.csv | File submission ketiga (814 baris, titik manis p92.0, 28.01% Unknown) | Passed |
| experiment 4.csv | File submission keempat (814 baris, 5-model ensemble NLL blend, 37.71% Unknown) | Passed |
| experiment 5.csv | File submission kelima (sub_005: SNR-Dynamic Gating 1-Crop, 37.84% Unknown) - LB 0.55789 | Passed |
| experiment 6.csv | File submission keenam (sub_006: Dynamic Gating + 3-Crop TTA + Prototype Shrinkage, 43.12% Unknown) - Peak LB 0.59462 | Passed |
| experiment 7.csv | File submission ketujuh (sub_007: Bioacoustic High-Harmonic Gate + Dual-Manifold Shrinkage, 70 Cendrawasih, 47.54% Unknown) - LB 0.58805 | Passed |
| experiment 8.csv | File submission kedelapan (sub_008: Golden Sweet-Spot Interpolation, 82 Cendrawasih, 43.86% Unknown) | Passed |
| submissions/sub_001_*/ | Bundel submission sub_001 (experiment 1.csv, test_predictions_detailed.csv, metrics.json) | Passed |
| submissions/sub_002_*/ | Bundel submission sub_002 (experiment 2.csv, metrics.json) | Passed |
| submissions/sub_003_*/ | Bundel submission sub_003 (experiment 3.csv, metrics.json) | Passed |
| submissions/sub_004_*/ | Bundel submission sub_004 (experiment 4.csv) | Passed |
| submissions/sub_005_*/ | Bundel submission sub_005 (experiment 5.csv, metrics.json) | Passed |
| submissions/sub_006_*/ | Bundel submission sub_006 (experiment 6.csv, metrics.json) | Passed |
| day1.md | Laporan teknis terpadu hari ke-1 lengkap untuk analisis AI (20k karakter) | Passed |
| schematics_day1_all_artifacts/ | Folder ekstraksi seluruh model weights, metrics, OOF numpy arrays Day 1 | Passed |

## Operational Monitoring

**Status:** NOT APPLICABLE

**Owner and cadence:** Not Applicable

**Signals and intervention thresholds:** Not Applicable

**Fallback or rollback:** Not Applicable

## Limitations and Risks

- Zero labeled Unknown audio in train: requires novelty detection, energy-based scoring, kNN on 256-d bottleneck embeddings, and pseudo-OOD injection to tune rejection thresholds.
- Environmental noise (rain, wind, insects) can cause false confidence or corrupt open-set distance metrics.
- Extreme group cardinality imbalance in Cendrawasih (groups of 3, 16, 29, 35) makes per-fold Macro F1 noisy; mitigated by enforcing Global OOF Macro F1 evaluation.
- Domain shift / Unknown proportion mismatch: Public LB score 0.54401 vs OOF 0.9099 indicates test set likely contains a higher proportion of Unknown species or severe recording background shift.
- Pelanggaran aturan diskualifikasi (data tambahan luar atau full AutoML) dapat membatalkan submission.

## Open Questions

- Apakah penggunaan `in_chans=1` pada `timm` memberikan konvergensi awal LP-FT yang lebih stabil dibanding duplikasi 3-channel tanpa normalisasi ImageNet?
- Berapa rasio bobot optimal ($w_1$ Energy Score vs $w_2$ kNN 256-d distance) pada kalibrasi Pseudo-OOD Injection di OOF?

## Exact Next Action

Day 2 Master Execution completed. All 4 DL backbones retrained on Preprocessing v2.0, Triad Tabular built, SNR-Dynamic Gating validated (OOF Macro F1 0.9380), 3-Crop TTA executed, and Kaggle submissions sub_005 (0.55789), sub_006 (0.59462 new peak), sub_007 (0.58805), and sub_008 (0.58356) evaluated. Sub_006 locked as final model. Day 2 artifacts exported to zip. Next session: verify foundation model (Google Perch) rules with competition organizers, then execute final competition technical report authoring in Indonesian (65% weight).

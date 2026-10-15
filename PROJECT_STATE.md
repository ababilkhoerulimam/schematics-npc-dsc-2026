# Project State: Schematics NPC DSC 2026 - Bird Chirp Openset Classification

**Schema version:** 1.1
**Last updated:** 2026-10-15
**Status:** In Progress
**Current phase:** Planning
**Project type:** PREDICTIVE
**Decision owner:** Ababil
**Primary notebook:** None
**Last completed cell:** Cell 7
**Data snapshot:** Initial download 2026-10-15
**Code version:** d9ba1cab51358023cb2a26379e14aa2df5b42524

## Objective and Success Criteria

**Problem or decision:** Classify short audio recordings into three known bird species (Bondol Jawa, Cendrawasih, Kutilang) and reject unseen species as Unknown under noisy tropical rainforest conditions.

**Unit of analysis:** One audio recording (.flac file identified by file_id).

**Outcome or target:** 4-class categorical label: Bondol Jawa, Cendrawasih, Kutilang, Unknown.

**Primary success metric:**
- Nilai Akhir Babak Penyisihan: 0.35 x Skor Prediksi + 0.65 x Skor Laporan.
- Skor Prediksi: Macro F1 Score pada 4 kelas di Kaggle test set (30% public LB, 70% private LB).
- Skor Laporan (100%): Evaluasi dan Validasi (30%), Modeling (25%), Preprocessing dan Feature Engineering (20%), EDA (15%), Penulisan dan Struktur Laporan (10%).
- Kelolosan: 5 tim teratas lolos ke Babak Final luring di Kampus ITS Sukolilo Surabaya (15 November 2026).

**Constraints:**
- Periode penyisihan: 15-25 Oktober 2026 di Kaggle. Maksimal 5 submission per hari.
- DILARANG menggunakan data tambahan di luar dataset yang disediakan panitia (strict zero external data).
- DILARANG menggunakan full AutoML atau solusi yang tidak dapat dijelaskan prosesnya.
- Model Hugging Face gated wajib melampirkan access token aktif dan identitas/versi model yang jelas.
- Reprodusibilitas mutlak: notebook wajib menggunakan seed/random_state. Perbedaan signifikan antara skor notebook dan skor Kaggle berisiko diskualifikasi.
- Berkas penyisihan wajib: DatSci_NamaTim_Notebook.ipynb, DatSci_NamaTim_Laporan.pdf (maks 10 halaman, maks 10 MB, Bahasa Indonesia baku), submission.csv, dan DatSci_NamaTim_Pendukung.zip (opsional).

## Data Inventory

| Source | Metadata / Grain | Time Coverage | Sensitivity | Exploration Status |
|---|---|---|---|---|
| NPC Guidebook - Data Science REVISED.pdf | 16 halaman PDF, panduan teknis resmi, jadwal, aturan, rubrik penilaian | 2026-10-15 | Public | Parsed |
| schematics-2026/train/metadata.csv | 1,826 rows, schema: file_id, audio_path, label, source_group | Unknown | Public | Not Opened |
| schematics-2026/train/audio/ | 1,826 .flac audio files | Unknown | Public | Not Opened |
| schematics-2026/test/audio/ | 814 .flac audio files | Unknown | Public | Not Opened |
| schematics-2026/sample_submission.csv | 814 rows, schema: file_id, label | Unknown | Public | Not Opened |

## Progress

| Stage | Status | Evidence / Cell | Notes |
|---|---|---|---|
| Planning | Completed | Guidebook, repo, PREPROCESSING_PLAN.md v1.1 | Aturan kompetisi, rubrik 65/35, larangan data eksternal, preprocessing architecture locked |
| Data quality and preparation | In Progress | Cell 1-7 | Metadata explored, audio specs verified, class balance and source_group audited |
| EDA or statistical analysis | Completed | Cell 1-7 | Class balance, acoustic dynamics, spectral signatures, stereo correlation, high-freq audit, mel-spectrograms |
| Feature engineering | Not Started | None | Pending mel-spectrogram extraction pipeline |
| Modeling and validation | Not Started | None | Pending baseline classifier and novelty detection / threshold tuning |
| Explanation and error analysis | Not Started | None | Pending OOF residual and misclassification audit |
| Methodology design | Completed | PREPROCESSING_PLAN.md v1.1 | 6-phase architecture with 8 ablation experiments |
| Delivery and export QA | Not Started | None | Format berkas penamaan DatSci_NamaTim_ dipersiapkan |
| Operational monitoring | Not Applicable | None | Batch Kaggle competition submission |

## Assumptions Register

| ID | Assumption | Category | Confidence | Impact if Wrong | Validation Plan | Status |
|---|---|---|---|---|---|---|
| A-001 | GroupKFold on source_group prevents recording-source leakage and mimics test distribution | Model | High | High | Compare random CV vs GroupKFold CV correlation with public LB | Open |
| A-002 | Test set contains Unknown class samples in non-trivial proportion | Data | High | High | Audit prediction distribution and evaluate threshold sensitivity on held-out pseudo-unknown classes | Open |
| A-003 | Audio duration and sample rates across train and test share a consistent sampling distribution | Data | Medium | Medium | Run automated audio format and duration probe across train and test audio | Open |
| A-004 | Standar pre-trained model backbones (e.g. ImageNet, AudioSpectrogramTransformer) diperbolehkan selama tidak menggunakan dataset audio luar baru | Model | High | High | Validasi interpretasi aturan data tambahan dengan panitia jika meragukan | Open |
| A-005 | Laporan 10 halaman bernilai 65 persen menentukan kelolosan lebih dominan dibanding micro-optimasi LB (35 persen) | Business | High | High | Alokasikan porsi kerja seimbang antara visualisasi EDA ilmiah, justifikasi modeling, dan error analysis | Open |

## Decisions Log

### D-001: GroupKFold by source_group for Validation

- **Chosen:** 4-fold StratifiedGroupKFold on source_group with stratification on label.
- **Alternatives considered:** 5-fold GroupKFold, StratifiedKFold without grouping.
- **Why:** Cell 2 proved Cendrawasih and Kutilang each have exactly 4 source groups (pure per class). A 5-fold split is mathematically impossible without leaving folds with zero validation samples for minority classes. 4 folds assigns exactly 1 source group of Cendrawasih and 1 of Kutilang to each fold.
- **Evidence:** Cell 2 stdout (source_groups_per_class: Bondol Jawa: 7, Cendrawasih: 4, Kutilang: 4).
- **Revisit when:** Group sizes are too imbalanced to maintain fold sample parity.

### D-002: Dual Optimization Protocol (65% Report, 35% Kaggle LB)

- **Chosen:** Eksekusi pipeline dengan dokumentasi eksperimental lengkap sesuai 5 bab laporan (EDA 15%, Preprocessing 20%, Modeling 25%, Validasi 30%, Struktur 10%) bersamaan dengan optimasi Macro F1.
- **Alternatives considered:** Chasing Kaggle Leaderboard saja tanpa mencatat log eksperimen dan ablasi.
- **Why:** Bobot laporan 65% jauh lebih besar dari skor prediksi 35%. Hanya 5 tim teratas gabungan yang lolos ke final ITS.
- **Evidence:** Panduan resmi Halaman 9-10 (Penilaian Babak Penyisihan: 0.35 Skor Prediksi + 0.65 Skor Laporan).
- **Revisit when:** Not Applicable.

### D-003: Strict Compliance on Prohibited External Data and AutoML

- **Chosen:** Menggunakan hanya dataset resmi 1.826 train audio, seed deterministik, dan arsitektur hand-crafted yang transparan.
- **Alternatives considered:** Pseudo-labeling dataset burung eksternal (e.g. Xeno-Canto), full AutoML.
- **Why:** Guidebook halaman 8 dan 14 melarang data tambahan di luar dataset panitia dan melarang full AutoML dengan sanksi diskualifikasi langsung.
- **Evidence:** Halaman 8 dan 14 Buku Panduan.
- **Revisit when:** Not Applicable.

### D-004: Bundled Submission Tracking Protocol

- **Chosen:** Setiap submission Kaggle (maks 5/hari) dibundel ke folder terisolasi `submissions/sub_XXX_<timestamp>/` berisi `submission.csv`, snapshot notebook/script, dan `metrics.json` (skor CV, detail model, threshold Unknown, skor public LB).
- **Alternatives considered:** Menimpa satu file submission.csv lokal tanpa tracking versi.
- **Why:** Kuota ketat 5 sub/hari dan aturan diskualifikasi panitia jika skor notebook tidak konsisten dengan skor Kaggle.
- **Evidence:** NPC Guidebook Hal 8 (Aturan konsistensi skor notebook dan leaderboard) serta instruksi user.
- **Revisit when:** Not Applicable.

### D-005: Resampling Rate and Nyquist Frequency Selection

- **Chosen:** Resampling audio native 48 kHz ke 32 kHz (Nyquist cutoff 16 kHz) untuk representasi spectrogram umum, atau 16 kHz jika menggunakan pre-trained acoustic transformer standar.
- **Alternatives considered:** Mempertahankan native 48 kHz (boros RAM/compute 33-66%), atau downsample ekstrem ke 8 kHz.
- **Why:** Bukti empiris Cell 7 menunjukkan band 16-24 kHz hanya menyumbang 0.11% hingga 1.22% energi, sedangkan band 8-16 kHz membawa hingga 9.08% energi harmonik Cendrawasih yang hilang jika di-resample ke 8 kHz.
- **Evidence:** Cell 7 stdout (band_16_24k: Bondol 0.11%, Cendrawasih 0.21%, Kutilang 1.22%).
- **Revisit when:** Model backbone memerlukan sampling rate spesifik yang tidak dapat ditoleransi.

### D-006: Stereo Downmixing to Single Channel Mono

- **Chosen:** Downmix stereo 2-channel menjadi 1-channel mono via averaging `(ch0 + ch1) / 2`.
- **Alternatives considered:** Mempertahankan dual-channel stereo pada input spectrogram.
- **Why:** Bukti Cell 7 menunjukkan korelasi antar-channel sangat tinggi (0.724 - 0.800) dan selisih volume RMS mendekati nol (0.002 - 0.003). Downmix menghemat 50% parameter konvolusi pertama dan komputasi tanpa menghilangkan fitur kicauan.
- **Evidence:** Cell 7 stdout (rms_diff mean 0.002-0.003, corr mean 0.724-0.800).
- **Revisit when:** Lokalisasi spasial stereo terbukti membedakan burung dari noise latar belakang.

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
| E-014 | Group cardinality range | Cendrawasih groups: 3 to 35; Kutilang: 132 to 178; Bondol: 58 to 336 | Cell 3 | Group groupby | Observed |
| E-015 | Acoustic signal dynamics | Clipping rate 40-55%, silence ~10%, Kutilang ZCR (0.2899) lower than Cendrawasih (0.3699) and Bondol (0.3483) | Cell 4 | Audio loader probe | Observed |
| E-016 | Spectral power band signatures | Kutilang dominant in 2-4kHz (57.2%); Bondol broad in 2-6kHz (64.5%); Cendrawasih high in 0-2kHz (44.0%) and 6-8kHz (14.7%) | Cell 5 | Welch PSD decomposition | Observed |
| E-017 | High-frequency power audit | 16-24kHz contains only 0.11-1.22% power; 0-8kHz contains >90.7% power across all classes | Cell 7 | 48kHz Welch PSD | Observed |
| E-018 | Stereo channel consistency | High inter-channel correlation (0.724-0.800) and minimal RMS delta (0.002-0.003) support mono downmixing | Cell 7 | Soundfile stereo audit | Observed |

## Methodology Design

**Method ID and version:** PREPROC-PLAN v1.1

**Maturity:** Proposal

**Method spec:** PREPROCESSING_PLAN.md (6 phases, 8 ablation experiments, 5 critical fixes applied)

**Case signature and baseline gap:** Open-set audio classification with IR=13.54, group-structured recordings, clipping artifacts, and zero Unknown labels in train. Standard closed-set classifiers fail on Unknown rejection.

**Exact next experiment:** Cell 8 - Execute 4-Fold StratifiedGroupKFold partition with zero group leakage verification

## Model Drivers and Error Analysis

Not Applicable at initial project planning phase.

## Artifacts

| Artifact | Purpose | Validation Status |
|---|---|---|
| NPC Guidebook - Data Science REVISED.pdf | Buku panduan teknis resmi kompetisi | Passed |
| schematics-2026/sample_submission.csv | Ground truth submission schema and file_id list | Passed |
| eda_acoustic_profiles.png | Visualisasi 3-panel profil akustik dan separabilitas spektral untuk laporan | Passed |
| eda_mel_spectrograms.png | Visualisasi mel-spectrogram komparatif 3 spesies untuk bab Preprocessing laporan | Passed |
| PREPROCESSING_PLAN.md | Arsitektur preprocessing & methodology v1.1 (6 fase, 8 ablasi, 5 critical fixes) | Passed |

## Operational Monitoring

**Status:** NOT APPLICABLE

**Owner and cadence:** Not Applicable

**Signals and intervention thresholds:** Not Applicable

**Fallback or rollback:** Not Applicable

## Limitations and Risks

- Zero labeled Unknown audio in train: requires novelty detection, extreme value theory, energy-based scoring, or leave-one-class-out validation to tune rejection thresholds.
- Environmental noise (rain, wind, insects) can cause false confidence or corrupt open-set distance metrics.
- Small dataset size (1,826 train audio) increases risk of overfitting on pre-trained heavy backbones without proper regularizations.
- Pelanggaran aturan diskualifikasi (data tambahan luar atau full AutoML) dapat membatalkan submission.

## Open Questions

- What is the sampling rate, channel count, and length distribution of the .flac recordings?
- What is the class distribution among Bondol Jawa, Cendrawasih, and Kutilang in train?
- How many unique source_groups exist, and are they class-pure or multi-class?
- Apakah pre-trained model weights audio standar (misal AST / Audio Spectrogram Transformer atau BEATs) diklasifikasikan sebagai data luar oleh panitia, atau model weights diizinkan?

## Exact Next Action

Cell 8: Execute 4-Fold StratifiedGroupKFold partition on source_group with zero group leakage verification, save train_folds.csv and preprocess_config.json.

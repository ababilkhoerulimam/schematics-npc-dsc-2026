# Analysis State: Bioacoustic Bird Call Dataset EDA & Diagnostic Analysis

**Schema version:** 1.0
**Last updated:** 2026-10-15
**Status:** Completed
**Current phase:** Synthesis
**Analysis type:** EXPLORATORY
**Decision owner / Stakeholder:** Jeremy Sion Anaru (Kaggle Bioacoustic Classification)
**Primary notebook / script:** Downloads/kaggle/train_audio_features.csv & test_audio_features.csv
**Last completed cell:** Cell 6
**Data snapshot:** C:\Users\HP\Downloads\kaggle

## Objective and Analytical Questions

**Primary business decision / question:** Menganalisis karakteristik bioakustik suara burung dalam data latih (Bondol Jawa, Cendrawasih, Kutilang) dan data uji, mendiagnosis potensi gangguan suara lingkungan (hujan, angin, serangga), serta merumuskan strategi deteksi Open-Set Recognition (OSR) untuk kelas `Unknown` pada data uji yang dievaluasi dengan metrik 4-Class Macro F1.

**Unit of analysis (Grain):** 1 file audio berformat FLAC (.flac) berdurasi ~5-15 detik beserta metadata rekaman (file_id, label kelas, source_group).

**Key metrics & definitions:**
- Macro F1 Score (4 Kelas): Rata-rata aritmatika unweighted dari F1 Bondol Jawa, Cendrawasih, Kutilang, dan Unknown ($0.25 \times \sum F1_c$).
- OOD Anomaly Distance: Jarak Euclidean terstandarisasi k-NN ($k=5$) dan skor Isolation Forest ke manifold data latih.
- Spectral Band Energy Ratio: Proporsi energi pada pita rendah (<1 kHz / noise lingkungan), menengah (1-4 kHz / Kutilang), dan tinggi (>4 kHz / Bondol Jawa).
- Sampling Rate & Channel Integrity: Standardisasi 48 kHz 24-bit stereo.
- Confounder Effect Size (Eta-squared): Besaran varians akustik yang terikat pada 15 klaster source_group.

**Scope & population boundaries:** 1,826 rekaman latih (3 kelas tertutup) dan 814 rekaman uji (open-set: 3 kelas + Unknown).

## Data Inventory

| Source | Metadata / Grain | Time Coverage | Sensitivity | Quality / Exploration Status |
|---|---|---|---|---|
| train/metadata.csv | 1,826 baris, 4 kolom (file_id, audio_path, label, source_group) | N/A | Public/Competition | Validated in Cell 1 (0 null, 100% unik) |
| train/audio/*.flac | 1,826 berkas audio FLAC 48 kHz stereo 24-bit | N/A | Public/Competition | Validated in Cell 2 (100% dapat di-decode) |
| test/audio/*.flac | 814 berkas audio FLAC 48 kHz stereo 24-bit | N/A | Public/Competition | Validated in Cell 2 (100% dapat di-decode) |
| sample_submission.csv | 814 baris, kolom file_id dan label | N/A | Public/Competition | Validated in Cell 1 (Cocok 100% dengan test set) |

## Progress & Cell Ledger

| Stage | Status | Cell / Source | Key Observed Evidence (Bold Numbers) |
|---|---|---|---|
| Data quality & grain validation | Completed | Cell 1 | **1,826** baris train metadata cocok 100% dengan audio train. **814** baris sample submission cocok 100% dengan audio test. **0** nilai null. |
| Audio encoding & signal properties | Completed | Cell 2 | **100%** berkas train (**1,826**) dan test (**814**) berformat FLAC stereo (2 kanal), sampling rate **48,000 Hz**, dan kedalaman bit **24-bit**. |
| Duration & temporal profiling | Completed | Cell 3 | Durasi train: mean **10.04s** (median **10.03s**, min **5.27s**, max **14.90s**). Durasi test: mean **10.09s** (median **10.09s**, min **5.55s**, max **14.90s**). KS-test $p = \mathbf{0.6695}$ (tidak ada drift durasi). |
| Spectral & acoustic separation | Completed | Cell 4 | Kutilang terkonsentrasi di mid-band 1-4 kHz (median **0.6405** vs Bondol Jawa **0.3761**, $p = \mathbf{1.26 \times 10^{-99}}$). Bondol Jawa dominan di high-band >4 kHz (median **0.4196**, frekuensi dominan **3,679.69 Hz**). Cendrawasih memiliki frekuensi dominan terendah (**1,992.19 Hz**) dan rolloff spektral terluas (**6,093.75 Hz**). |
| Simpson's paradox & confounder audit | Completed | Cell 5 | Seluruh **15** source_group 100% nested dalam satu kelas (0 overlap antar spesies). Source group menjelaskan hingga $\mathbf{23.7\%}$ varians akustik dalam spesies yang sama ($p = \mathbf{1.86 \times 10^{-55}}$). Membutuhkan Stratified Group K-Fold. |
| Open-Set & Unknown OOD Profiling | Completed | Cell 6 | Teridentifikasi **101 dari 814** sampel test (**12.41%**) berada di luar ambang P95 jarak manifold data latih. Frekuensi dominan klaster OOD anjlok ke median **796.88 Hz** dan low-band noise melonjak ke **0.2199** (indikasi hujan/angin/spesies asing). |

## Hypotheses & Findings Ledger

| ID | Type | Statement | Evidence / Cell | Status |
|---|---|---|---|---|
| E-001 | Finding | Train set terdiri dari 1,826 audio dan Test set terdiri dari 814 audio, seluruhnya terbaca sempurna tanpa berkas rusak. | Cell 1 & 2 | Robust |
| E-002 | Finding | Distribusi kelas latih mengalami ketimpangan moderat hingga ekstrem: Bondol Jawa 1,124 (61.56%), Kutilang 619 (33.90%), dan Cendrawasih 83 (4.55%), rasio perbandingan 13.5 : 7.5 : 1. | Cell 1 | Robust |
| E-003 | Finding | Parameter teknis audio bersifat 100% seragam: 48,000 Hz sample rate, 24-bit bit depth, dan 2 kanal stereo. 97.04% train dan 96.56% test merupakan true stereo dengan diferensiasi kanal aktif, sedangkan sisanya dual-mono. | Cell 2 | Robust |
| E-004 | Finding | Durasi audio dibatasi dalam rentang ~5.27s hingga 14.90s dengan median 10.03s. Tidak terdeteksi pergeseran domain temporal antara train dan test (KS-test stat 0.0302, p-value 0.6695). | Cell 3 | Robust |
| E-005 | Finding | Karakteristik bioakustik ketiga spesies terpisah secara signifikan pada spektrum frekuensi: Kutilang unggul pada pita menengah (Kruskal-Wallis H=455.45), Bondol Jawa pada pita tinggi (H=391.45), dan Cendrawasih pada frekuensi resonansi fundamental rendah (median 1,992.19 Hz). | Cell 4 | Robust |
| E-006 | Finding | Terdapat 15 source_group yang 100% terisolasi per spesies (7 pada Bondol Jawa, 4 pada Kutilang, 4 pada Cendrawasih). Variabel ini merepresentasikan klaster sesi/alat perekaman yang menyumbang varians akustik internal signifikan. | Cell 5 | Robust |
| E-007 | Finding | Sebanyak 101 sampel uji (12.41%) menyimpang dari manifold training (>P95 dist) dengan konsentrasi energi pita rendah (<1 kHz) sebesar 22.0% (vs 10.4% pada train), memvalidasi keberadaan kelas Unknown dari derau lingkungan atau spesies tak dikenal. | Cell 6 | Robust |
| H-001 | Hypothesis | Evaluasi validasi dengan random split biasa akan mengalami data leakage parah akibat pengelompokan segmen audio dari berkas perekaman asal yang sama (source_group). | Cell 5 | Supported |
| H-002 | Hypothesis | Model tertutup tanpa mekanisme penolakan Unknown akan kehilangan 25% bobot Macro F1 secara otomatis. | Deskripsi Kompetisi | Supported |

## Key Decisions & Assumptions

| Date | Decision / Metric Rule | Evidence / Rationale | Impact on Conclusions |
|---|---|---|---|
| 2026-10-15 | Formulasi evaluasi 4-Class Macro F1 | Dokumen resmi kompetisi menetapkan bobot seimbang 25% untuk Bondol Jawa, Kutilang, Cendrawasih, dan Unknown | Critical |
| 2026-10-15 | Ekstraksi fitur DSP berbasis Welch Periodogram (nperseg=2048) | Karakteristik bioakustik burung stabil pada estimasi densitas daya spektral resolusi frekuensi ~23.4 Hz | High |
| 2026-10-15 | Verifikasi true stereo vs dual-mono | Mendeteksi variasi perekaman mikrofon ganda vs duplikasi mono | Medium |
| 2026-10-15 | Rekomendasi Stratified Group K-Fold berdasarkan source_group | Menghindari group leakage dari 15 klaster perekaman | Critical |
| 2026-10-15 | Strategi deteksi Unknown via pseudo-OOD validation | Kalibrasi ambang batas penolakan kelas pada CV untuk memaksimalkan Macro F1 4-kelas | Critical |

## Blockers, Limitations & Next Analytical Step

- **Data limitations:** Spesies Cendrawasih memiliki jumlah sampel terbatas (83 sampel latih yang berasal dari hanya 4 sesi source_group), sementara kelas Unknown tidak memiliki data latih langsung.
- **Current blocker:** Tidak ada. Seluruh data latih dan uji berhasil diekstraksi dan dianalisis secara komprehensif.
- **Exact next step:** Menyusun blueprint arsitektur pemodelan bioakustik, strategi pseudo-Unknown validation, dan kalibrasi ambang batas prediksi untuk kompetisi.

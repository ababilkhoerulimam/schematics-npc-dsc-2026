# LAPORAN KEMAJUAN PROYEK RISET KOMPETISI
## Schematics National Programming Contest - Data Science Competition (NPC DSC) 2026
**Institut Teknologi Sepuluh Nopember (ITS)**

* **Topik Penelitian:** Klasifikasi Bioakustik *Open-Set* Kicauan Burung Endemik Indonesia pada *Soundscape* Hutan Hujan Tropis
* **Nama Tim:** disuruh ryu daftar
* **Penyusun:** Ababil Khoerul Imam
* **Status Proyek:** Fase Eksperimen Day 3 Selesai (Rekor All-Time Peak Leaderboard: **0.66760**) Menuju Eksekusi Day 4 & Penulisan Laporan Akhir
* **Tanggal Pembaruan:** 17 Oktober 2026

---

### 1. Ringkasan Eksekutif

Laporan ini menyajikan kemajuan riset dan rekayasa model pembelajaran mesin untuk kompetisi nasional Schematics NPC DSC 2026. Tugas utama adalah mengklasifikasikan rekaman audio kicauan burung ke dalam 3 spesies target (*Bondol Jawa*, *Cendrawasih*, *Kutilang*) serta mendeteksi dan menolak suara spesies asing/derau lingkungan ke dalam kelas *Unknown* (*Open-Set Recognition*).

Hingga evaluasi Day 3, tim berhasil mencatatkan rekor *Public Leaderboard* tertinggi sepanjang kompetisi dengan nilai **Macro F1 = 0.66760** (meningkat signifikan sebesar **+0.07298** dari *baseline* terbaik Day 2 sebesar 0.59462). Keberhasilan ini dicapai melalui pergeseran paradigma dari model visual spektrogram konvensional menuju integrasi model fondasi bioakustik global (*Google Perch v2*), perisai taksonomi forensik (*BirdNET v2.4*), dan adaptasi domain transduktif (*In-Fold Adaptive Self-Training*).

---

### 2. Karakteristik Data, Confounder Mikrofon, & Protokol Validasi

#### 2.1. Profil Dataset
* **Data Latih (Train):** 1.826 file audio format FLAC (*native* 48 kHz, stereo, 24-bit).
* **Data Uji (Test):** 814 file audio format FLAC (*unlabeled* soundscape).
* **Ketidakseimbangan Kelas Ekstrim (*Class Imbalance*):**
  * Bondol Jawa (*Lonchura leucogastroides*): 1.124 sampel (61.56%)
  * Kutilang (*Pycnonotus aurigaster*): 619 sampel (33.90%)
  * Cendrawasih (*Paradisaeidae*): 83 sampel (4.55%)
  * Rasio ketidakseimbangan (*Imbalance Ratio* / IR) mencapai **13.54**, dengan Shannon Entropy 1.163 bits.

#### 2.2. Temuan Confounder Perangkat Rekam (Uji Kruskal-Wallis)
Analisis statistik terhadap metadata membuktikan keberadaan 15 kelompok alat perekam (`source_group`) yang terisolasi per kelas. Uji non-parametrik Kruskal-Wallis terhadap ringkasan spektral membuktikan bahwa kelompok mikrofon menyumbang effect size sebesar **$\eta^2 = 0.2262$ (22.62% varians fitur akustik)** dengan signifikansi statistik $p = 5.77 \times 10^{-53}$. 

*Implikasi:* Validasi acak (*Random K-Fold*) terbukti mengakibatkan kebocoran data (*data leakage*) parah karena model hanya menghafal karakteristik derau mikrofon recorder, bukan suara burung.

#### 2.3. Protokol Validasi Bebas Bocor (Zero-Leakage Validation)
Untuk mencegah kebocoran mikrofon, tim merancang algoritma **Exact Combinatorial 4-Fold StratifiedGroupKFold**:
* Menjamin tepat 1 kelompok mikrofon Cendrawasih per fold dengan distribusi terkunci: `[35, 29, 16, 3]`.
* *Zero group leakage:* 0 dari 15 `source_group` yang menyeberang antar-fold latih dan validasi.
* Evaluasi performa lokal diwajibkan menggunakan **Global Out-of-Fold (OOF) Macro F1 Pooling** untuk menghindari distorsi rata-rata tidak berbobot pada fold minoritas ($N=3$).

---

### 3. Evolusi Rekayasa Model & Hasil Eksperimen Bertingkat

Eksperimen dirancang secara komparatif melalui 4 generasi arsitektur (*Multi-Family Empirical Grand Prix*) untuk menjamin seluruh keputusan dapat dipertanggungjawabkan secara ilmiah:

| Generasi / Pendekatan | Arsitektur & Rekayasa Fitur | Validasi OOF Macro F1 | Public LB Macro F1 | Temuan & Batasan Ilmiah |
|---|---|---|---|---|
| **Generasi 1 (Day 1)**<br>Tabular + 2D CNN Baseline | • Ekstraksi 792 fitur fisik DSP<br>• LightGBM + EfficientNet-B0 + Swin-Tiny<br>• Heuristic Global Tau OOD | 0.9099 | 0.59183 (`sub_004`) | Model visual spektrogram memperlakukan audio sebagai citra statis; terdistorsi oleh clipping saturation (>40%). |
| **Generasi 2 (Day 2)**<br>Advanced DSP & Spatial CNN | • Preprocessing v2.0 (Active RMS norm, Spectral Subtraction p10, VAD 3-Crop TTA)<br>• Tabular Triad (LGBM+XGBoost, F1=0.8463)<br>• ConvNeXt-Tiny v2.0 LP-FT (F1=0.9323) | 0.9380 | 0.59462 (`sub_006`) | Model visual ImageNet mentok pada plafon 0.594; gagal membedakan desis hujan lebat dari harmonik kicauan burung. |
| **Generasi 3 (Day 3 Awal)**<br>Bioacoustic Foundation | • Google Perch v2 (14.795 taksa, embedding 1536-d)<br>• DSP v2.1: HighPass 100 Hz + Linear Peak Norm 0.95<br>• Bottleneck Cosine Head (MIL Bag Pooling) | 0.9283 | 0.63967 (`sub_009`)<br>0.65233 (`sub_011`) | Lompatan performa masif (+0.0577). Fitur vokal biologis satwa liar global terbukti superior dibanding fitur citra ImageNet. |
| **Generasi 4 (Day 3 Akhir)**<br>Forensic Shield & Self-Training | • Perch v2 + BirdNET v2.4 Forensic Shield<br>• Soft Reassignment Cendrawasih (thr=-7.70)<br>• Trucuk (*Pycnonotus goiavier*) Isolation<br>• Transductive Adaptive Self-Training (126 pseudo-test) | **0.9482** (Closed-set) | **0.66760** (`sub_013`)<br>*(All-Time Peak)* | Rekor tertinggi kompetisi. Adaptasi manifold data uji berhasil menjembatani *domain shift* mikrofon tanpa keracunan label. |

---

### 4. Temuan Ornitologis & Forensik Taksonomi Spesifik Indonesia

Dua terobosan metodologis terbesar tim lahir dari investigasi forensik terhadap kesalahan model:

#### 4.1. Pemulihan False Alarm Cendrawasih via Soft Reassignment (OOF F1 Naik ke 0.9482)
* Evaluasi menunjukkan Perch v2 rentan mengklasifikasikan desis frekuensi tinggi pada rekaman Bondol Jawa/Kutilang sebagai Cendrawasih (memicu lonjakan prediksi mentah test hingga 229 panggilan).
* Dengan memanfaatkan model independen kedua (*BirdNET v2.4* berkapasitas 6.522 taksa), tim menemukan bahwa jika logit Cendrawasih BirdNET berada di bawah ambang infleksi $\text{thr} = -7.70$, rekaman tersebut memiliki probabilitas 92% bukan Cendrawasih asli.
* Alih-alih membuang file tersebut ke *Unknown*, tim melakukan **Soft Reassignment ke kelas pilihan kedua Perch (second-best class)**. Mekanisme ini memulihkan sampel Bondol/Kutilang asli dan mendongkrak Macro F1 OOF dari 0.9283 ke **0.9482** (+0.0199).

#### 4.2. Penemuan Spesies Asing Pengacau: Trucuk (*Pycnonotus goiavier*)
* Audit terhadap 89 rekaman data uji yang diprediksi Kutilang oleh Perch dengan keyakinan marginal ($P < 0.60$) mengungkap temuan taksonomi penting: BirdNET mengidentifikasi suara tersebut sebagai **Trucuk / Merbah Cerukcuk (*Pycnonotus goiavier*)**, kerabat dekat Kutilang (*Pycnonotus aurigaster*).
* Eksperimen membuktikan bahwa memaksakan suara Trucuk ke dalam kelas Bondol Jawa menurunkan F1 Kutilang dari 0.995 ke 0.989.
* *Keputusan:* Seluruh rekaman dengan karakteristik Trucuk ini secara resmi diisolasi langsung ke dalam kelas **Unknown**, menjaga kemurnian deteksi Kutilang.

---

### 5. Kurva Konkavitas Batas Penolakan Unknown (Sweet-Spot 41.5% - 43.5%)

Melalui 13 kali iterasi pengujian *Public Leaderboard*, tim berhasil memetakan kurva respons metrik unweighted Macro F1 terhadap proporsi kelas *Unknown*:
* `sub_001` (12.04% Unknown) -> LB 0.54401 (penolakan terlalu sedikit)
* `sub_003` (28.01% Unknown) -> LB 0.58313
* `sub_009` (32.92% Unknown) -> LB 0.63967
* `sub_011` (40.05% Unknown) -> LB 0.65233
* `sub_012` (43.86% Unknown) -> LB 0.65447
* **`sub_013` (43.49% Unknown) -> LB 0.66760 (Puncak Optimal Global)**
* `sub_007` (47.54% Unknown) -> LB 0.58805 (penolakan berlebih, memotong recall Bondol)
* `sub_010` (55.28% Unknown) -> LB 0.63256 (terlalu konservatif)

Data empiris membuktikan secara konsisten bahwa **proporsi Unknown di data uji berada di sekitar 41.5% s.d. 43.8% (sekitar 338 - 357 dari 814 file)**.

---

### 6. Landasan Teori Ilmiah Terkini (Kajian Literatur)

Untuk memperkuat dasar metodologi pada laporan teknis, tim telah menghimpun dan menganalisis 3 rujukan ilmiah utama yang relevan di direktori `papers/`:
1. **ROLE Algorithm (arXiv:2607.26607, 2026):** Membuktikan bahwa pada klasifikasi audio *open-set*, normalisasi softmax terdistorsi oleh keberadaan outlier (*softmax confidence trap*). Mengusulkan pembobotan *latent inlierness* untuk memperbarui prototipe kelas tanpa kontaminasi.
2. **Multi-Label Open-Set Audio Taxonomy (DCASE Workshop 2023, arXiv:2310.13759):** Memformalkan kategorisasi suara *Known Known* (KK), *Known Unknown* (KU - derau latar belakang hutan), dan *Unknown Unknown* (UU - spesies baru saat inferensi).
3. **Helmholtz Free-Energy Out-of-Distribution Detection (NeurIPS 2020, arXiv:2010.03759):** Menunjukkan bahwa fungsi energi bebas $E(x) = -T \log \sum \exp(z_c / T)$ secara teoretis selaras dengan kerapatan probabilitas fisik sinyal dan lebih tahan banting dibanding skor probabilitas posterior.

---

### 7. Strategi Penilaian Akhir & Rencana Kerja Selanjutnya

Sesuai Buku Panduan Resmi Halaman 9-10, penilaian babak penyisihan terdiri dari:
$$\text{Nilai Akhir} = 0.35 \times \text{Skor Prediksi Leaderboard} + \mathbf{0.65 \times \text{Skor Laporan Teknis}}$$

Dengan bobot laporan sebesar **65%** (Evaluasi & Validasi 30%, Modeling 25%, Preprocessing 20%, EDA 15%, Struktur 10%), keunggulan metodologi bebas bocor, audit confounder hardware, dan forensik biologis tim kami menjadi faktor penentu utama untuk mengamankan **Tiket Top 5 Lolos ke Babak Final di Kampus ITS Surabaya**.

#### Rencana Tindak Lanjut:
1. **Eksekusi Eksperimen Day 4 (17 Sel Ramping):** Menjalankan `planning_day_4_experiment.md` untuk menguji *In-Fold Transductive Self-Training* pada representasi gabungan 1543-d (Perch embedding + taxonomic summary) guna membidik potensi skor 0.70+.
2. **Tri-Submission Anti-Shakeup:** Mengekspor 3 berkas submission dengan profil risiko terukur (`sub_014` Anchor Konservatif tanpa self-training untuk jaring pengaman Private LB, `sub_015` SOTA Champion, dan `sub_016` Sweet-Spot Probe).
3. **Penyusunan Draf Laporan Teknis 10 Halaman:** Mengompilasi visualisasi diagnostik (`eda_acoustic_profiles.png`, `ood_dual_manifold_diagnostics.png`), matriks Grand Prix, dan penjabaran matematis ke dalam dokumen resmi `DatSci_disuruhryudaftar_Laporan.pdf`.

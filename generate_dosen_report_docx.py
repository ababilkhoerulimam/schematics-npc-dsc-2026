import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_footer_page_number(run):
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w'))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls('w'))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls('w'))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w'))
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

def create_dosen_report_docx(output_path="Laporan_Kemajuan_Schematics_2026_Dosen.docx"):
    doc = Document()

    # Page Margins: 0.8 inch all around
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

        # Header
        header = s.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("SCHEMATICS NPC DSC 2026 | Laporan Kemajuan Riset Bioakustik Open-Set")
        hrun.font.name = "Arial"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(113, 128, 150)

        # Footer
        footer = s.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
        frun1 = fp.add_run("Institut Teknologi Sepuluh Nopember (ITS) | Tim: disuruh ryu daftar (UNESA)")
        frun1.font.name = "Arial"
        frun1.font.size = Pt(8.5)
        frun1.font.color.rgb = RGBColor(113, 128, 150)
        
        frun_mid = fp.add_run("\t\tHalaman ")
        frun_mid.font.name = "Arial"
        frun_mid.font.size = Pt(8.5)
        frun_mid.font.color.rgb = RGBColor(113, 128, 150)
        add_footer_page_number(frun_mid)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    p_title.paragraph_format.line_spacing = 1.15
    run_t = p_title.add_run("LAPORAN KEMAJUAN RISET & REKAYASA MODEL PREDIKSI")
    run_t.font.name = "Arial"
    run_t.font.size = Pt(15)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(15, 41, 66) # Deep Navy

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(10)
    run_sub = p_sub.add_run("Kompetisi Sains Data Nasional - Schematics NPC Data Science Competition (DSC) 2026")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(10.5)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(30, 78, 121) # Academic Blue

    # Metadata Box Table
    meta_table = doc.add_table(rows=3, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    col_widths = [Inches(1.2), Inches(2.3), Inches(1.1), Inches(2.3)]
    for row in meta_table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = width

    meta_entries = [
        ("Topik Riset", "Klasifikasi Bioakustik Open-Set Kicauan Burung pada Soundscape Hutan Tropis",
         "Penyusun", "Ababil Khoerul Imam (S1 Sains Data UNESA)"),
        ("Nama Tim", "disuruh ryu daftar (ID Tim Kompetisi Resmi)",
         "Pencapaian", "All-Time Peak LB: 0.66760 (Peringkat 14 Nasional)"),
        ("Penyelenggara", "Institut Teknologi Sepuluh Nopember (ITS) Surabaya",
         "Bobot Skor", "65% Laporan Teknis  +  35% Skor Prediksi Kaggle")
    ]

    for r_idx, (k1, v1, k2, v2) in enumerate(meta_entries):
        row = meta_table.rows[r_idx]
        for c in row.cells:
            set_cell_background(c, "F8FAFC")
            set_cell_margins(c, top=80, bottom=80, left=100, right=100)
        
        # cell 0: k1
        p = row.cells[0].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(k1)
        r.font.name = "Arial"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(15, 41, 66)

        # cell 1: v1
        p = row.cells[1].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(v1)
        r.font.name = "Arial"
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(45, 55, 72)

        # cell 2: k2
        p = row.cells[2].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(k2)
        r.font.name = "Arial"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(15, 41, 66)

        # cell 3: v2
        p = row.cells[3].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(v2)
        r.font.name = "Arial"
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(45, 55, 72)
        if "0.66760" in v2 or "65%" in v2:
            r.font.bold = True

    # Helper for adding headings
    def add_sec_heading(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(title)
        run.font.name = "Arial"
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(15, 41, 66)
        return p

    def add_body_p(text, bold_prefix=None, space_after=4):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = "Arial"
            r_pre.font.size = Pt(9.5)
            r_pre.font.bold = True
            r_pre.font.color.rgb = RGBColor(30, 41, 59)
        r_text = p.add_run(text)
        r_text.font.name = "Arial"
        r_text.font.size = Pt(9.5)
        r_text.font.color.rgb = RGBColor(45, 55, 72)
        return p

    def add_bullet_p(title_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        r_pre = p.add_run(title_prefix)
        r_pre.font.name = "Arial"
        r_pre.font.size = Pt(9.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(30, 41, 59)
        r_text = p.add_run(text)
        r_text.font.name = "Arial"
        r_text.font.size = Pt(9.5)
        r_text.font.color.rgb = RGBColor(45, 55, 72)
        return p

    # 1. Ringkasan Eksekutif
    add_sec_heading("1. Ringkasan Eksekutif & Objektif Utama")
    add_body_p(
        "Laporan ini mendokumentasikan rekayasa sistem kecerdasan artifisial bioakustik dalam kompetisi nasional Schematics NPC DSC 2026 yang diselenggarakan oleh ITS Surabaya. "
        "Tugas utama adalah mengenali 3 spesies burung target (Bondol Jawa, Cendrawasih, Kutilang) dan mendeteksi sekaligus menolak spesies asing atau derau latar belakang ke dalam kelas Unknown (Open-Set Recognition). "
        "Melalui serangkaian eksperimen terukur hingga Day 3, tim berhasil memecahkan rekor performa dengan skor Public Leaderboard Macro F1 = 0.66760 (lonjakan +0.07298 dari rekor visual Day 2 sebesar 0.59462). "
        "Keberhasilan ini didorong oleh pergeseran paradigma dari model visual spektrogram konvensional menuju integrasi model fondasi Google Perch v2 (14.795 taksa), perisai taksonomi BirdNET v2.4, dan adaptasi domain transduktif (In-Fold Adaptive Self-Training)."
    )

    # 2. Karakteristik Data
    add_sec_heading("2. Karakteristik Data, Confounder Perangkat Rekam, & Protokol Validasi Bebas Bocor")
    add_body_p(
        "Dataset terdiri atas 1.826 audio latih dan 814 audio uji (format FLAC stereo 48 kHz 24-bit). Distribusi kelas latih mengalami ketimpangan ekstrim (Imbalance Ratio = 13.54, Shannon Entropy = 1.163 bits): Bondol Jawa sebanyak 1.124 sampel (61.56%), Kutilang 619 sampel (33.90%), dan Cendrawasih hanya 83 sampel (4.55%).",
        bold_prefix="2.1 Profil Dataset & Ketidakseimbangan Ekstrim: "
    )
    add_body_p(
        "Metadata mengungkap rekaman berasal dari 15 kelompok perangkat perekam (source_group) yang terisolasi per kelas. Pengujian hipotesis non-parametrik Kruskal-Wallis membuktikan bahwa kelompok mikrofon menyumbang effect size sebesar eta-squared = 0.2262 (22.62% varians akustik) dengan signifikansi statistik sangat tinggi (H = 272.52, p = 5.77 x 10^-53). Hal ini membuktikan bahwa validasi acak biasa (Random K-Fold) dipastikan mengalami kebocoran data berat (data leakage) karena model hanya menghafal karakteristik mikrofon.",
        bold_prefix="2.2 Bukti Statistik Confounder Mikrofon (Uji Kruskal-Wallis): "
    )
    add_body_p(
        "Untuk menjamin integritas saintifik, dirancang partisi kombinatorial eksak yang memetakan 15 kelompok perekam ke 4 fold tanpa kebocoran (zero group leakage) dengan distribusi Cendrawasih terkunci: [35, 29, 16, 3]. Evaluasi performa lokal diwajibkan menggunakan Global Out-of-Fold (OOF) Macro F1 Pooling pada seluruh 1.826 prediksi untuk menghindari distorsi rata-rata fold minoritas.",
        bold_prefix="2.3 Partisi Bebas Bocor (Exact Combinatorial 4-Fold GroupKFold): "
    )

    # 3. Evolusi Rekayasa Model
    add_sec_heading("3. Evolusi Rekayasa Model & Matriks Pembuktian Empiris (Day 1 s.d. Day 3)")
    add_body_p("Seluruh tahapan eksperimen dirancang modular melalui 4 generasi pemodelan dalam protokol evaluasi yang identik:")

    # Table 1: Generation comparison
    t1 = doc.add_table(rows=5, cols=5)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1.autofit = False

    t1_widths = [Inches(1.2), Inches(2.2), Inches(0.7), Inches(0.9), Inches(1.9)]
    for row in t1.rows:
        for idx, w in enumerate(t1_widths):
            row.cells[idx].width = w

    headers_t1 = ["Generasi Model", "Arsitektur & Rekayasa Fitur", "OOF F1", "Public LB", "Temuan & Implikasi Ilmiah"]
    for idx, text in enumerate(headers_t1):
        cell = t1.rows[0].cells[idx]
        set_cell_background(cell, "0F2942")
        set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx in [2, 3] else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    data_t1 = [
        ("Generasi 1 (Day 1)\nTabular + 2D CNN",
         "792 fitur DSP tabular + LightGBM + EfficientNet-B0 + Swin-Tiny. Heuristic Global Tau p95.",
         "0.9099", "0.59183\n(sub_004)",
         "Pondasi visual spektrogram 2D; performa terdistorsi oleh clipping saturation hardware (>40%)."),
        ("Generasi 2 (Day 2)\nSpatial CNN + Triad",
         "Preprocessing v2.0 (Active RMS norm, spectral subtraction p10). ConvNeXt-Tiny v2.0 LP-FT 2-phase + 3-Crop TTA + Tabular Triad (LGBM+XGB).",
         "0.9380", "0.59462\n(sub_006)",
         "Plafon maksimal model visual ImageNet; model gagal membedakan derau hujan hutan lebat dari vokal kicauan."),
        ("Generasi 3 (Day 3)\nBioacoustic Foundation",
         "Google Perch v2 (14.795 taksa dunia, 1536-d). DSP v2.1: Highpass 100Hz + Linear Peak Norm 0.95 (tanpa tanh). Bottleneck Cosine Head 256-d.",
         "0.9283", "0.65233\n(sub_011)",
         "Lompatan performa masif (+0.0577). Fitur vokal biologis satwa liar dunia terbukti superior dibanding fitur citra umum."),
        ("Generasi 4 (Day 3)\nForensic + Self-Training",
         "Perch v2 + BirdNET v2.4 (6.522 taksa) Forensic Dual-Shield (reassign Cendrawasih bising + isolasi Trucuk) + Adaptive Self-Training (126 pseudo-test).",
         "0.9482", "0.66760\n(sub_013)",
         "Rekor tertinggi kompetisi. Adaptasi domain data uji berhasil menjembatani gap akustik mikrofon tanpa keracunan label.")
    ]

    for r_idx, row_data in enumerate(data_t1, start=1):
        row = t1.rows[r_idx]
        bg = "FFFFFF" if r_idx % 2 == 1 else "F8FAFC"
        for c_idx, text in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [2, 3] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(text)
            r.font.name = "Arial"
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(45, 55, 72)
            if r_idx == 4 and c_idx in [2, 3]:
                r.font.bold = True
                r.font.color.rgb = RGBColor(15, 41, 66)

    # 4. Temuan Ornitologis
    add_sec_heading("4. Temuan Ornitologis & Forensik Taksonomi Spesifik Indonesia")
    add_body_p("Dua terobosan metodologis terbesar tim yang mengangkat performa model berakar pada penyelidikan forensik biologis:")
    add_body_p(
        "Evaluasi membuktikan Perch v2 rentan mengklasifikasikan desis frekuensi tinggi pada rekaman Bondol Jawa/Kutilang sebagai Cendrawasih (memicu lonjakan 229 prediksi Cendrawasih mentah di data uji). Melalui integrasi model independen kedua (BirdNET v2.4 berkapasitas 6.522 taksa), ditemukan bahwa jika logit BirdNET Cendrawasih < -7.70 (ambang infleksi Youden), rekaman tersebut memiliki reliabilitas 92% bukan Cendrawasih asli. Mengembalikan rekaman ini ke pilihan kedua Perch (second-best class) terbukti memulihkan sampel Bondol/Kutilang asli dan mendongkrak Macro F1 OOF dari 0.9283 menjadi 0.9482 (+0.0199).",
        bold_prefix="4.1 Pemulihan False Alarm Cendrawasih via Soft Reassignment (OOF F1 Naik ke 0.9482): "
    )
    add_body_p(
        "Pemeriksaan taksonomi pada 89 rekaman data uji yang diprediksi Kutilang oleh Perch dengan keyakinan marjinal (P < 0.60) mengungkap bahwa suara tersebut merupakan Trucuk / Merbah Cerukcuk (Pycnonotus goiavier), spesies kerabat dekat Kutilang (Pycnonotus aurigaster). Eksperimen membuktikan bahwa memaksakan suara Trucuk ke dalam kelas Bondol Jawa menurunkan F1 Kutilang dari 0.995 ke 0.989. Solusi defensibel: seluruh 72 rekaman Trucuk ini diisolasi langsung ke dalam kelas Unknown, mengamankan kemurnian presisi kelas Kutilang.",
        bold_prefix="4.2 Identifikasi Spesies Asing Pengacau: Trucuk (Pycnonotus goiavier): "
    )

    # 5. Kurva Konkavitas
    add_sec_heading("5. Karakterisasi Kurva Konkavitas Batas Penolakan Unknown (Sweet-Spot)")
    add_body_p(
        "Melalui 13 kali pengujian terukur di Public Leaderboard, tim berhasil memetakan kurva respons metrik Macro F1 terhadap proporsi kuantil penolakan kelas Unknown. "
        "Data empiris membuktikan secara konsisten bahwa populasi riil kelas Unknown berada pada rentang 41.5% s.d. 43.8% (338 - 357 dari 814 rekaman uji):"
    )

    # Table 2: Unknown Sweet-Spot Curve
    t2 = doc.add_table(rows=9, cols=6)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.autofit = False

    t2_widths = [Inches(0.9), Inches(1.6), Inches(0.8), Inches(0.9), Inches(0.7), Inches(2.0)]
    for row in t2.rows:
        for idx, w in enumerate(t2_widths):
            row.cells[idx].width = w

    headers_t2 = ["ID Sub", "Arsitektur Inti", "Unknown %", "Jumlah Unk", "Public LB", "Dampak / Fenomena Evaluasi"]
    for idx, text in enumerate(headers_t2):
        cell = t2.rows[0].cells[idx]
        set_cell_background(cell, "1E4E79") # Academic Blue
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx in [2, 3, 4] else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    data_t2 = [
        ("sub_001", "Tabular + 2D CNN", "12.04%", "98 / 814", "0.54401", "Penolakan terlalu sedikit; membiarkan banyak suara asing masuk."),
        ("sub_003", "Threshold Probe p92", "28.01%", "228 / 814", "0.58313", "Peningkatan linear; zona transisi pemisahan domain."),
        ("sub_009", "Google Perch v2 Pure", "32.92%", "268 / 814", "0.63967", "Lompatan bioakustik; batas bawah Unknown mulai terdefinisi."),
        ("sub_011", "Perch v2 Calibrated", "40.05%", "326 / 814", "0.65233", "Memasuki dataran tinggi sweet-spot kompetisi."),
        ("sub_012", "Perch + BirdNET Shield", "43.86%", "357 / 814", "0.65447", "Presisi Cendrawasih pulih; Trucuk tersaring ke Unknown."),
        ("sub_013", "Perch + Self-Training", "43.49%", "354 / 814", "0.66760", "Puncak Optimal Global; konvergensi presisi dan recall tertinggi."),
        ("sub_007", "Over-rejection Bound", "47.54%", "387 / 814", "0.58805", "Penolakan berlebih; mulai memotong sampel Bondol Jawa asli."),
        ("sub_010", "Conservative Multi-OOD", "55.28%", "450 / 814", "0.63256", "Terlalu konservatif; F1 anjlok akibat recall kelas target turun.")
    ]

    for r_idx, row_data in enumerate(data_t2, start=1):
        row = t2.rows[r_idx]
        bg = "FFFFFF" if r_idx % 2 == 1 else "F8FAFC"
        is_peak = (r_idx == 6)
        for c_idx, text in enumerate(row_data):
            cell = row.cells[c_idx]
            set_cell_background(cell, "FEF3C7" if is_peak else bg) # highlight peak row
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx in [2, 3, 4] else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(text)
            r.font.name = "Arial"
            r.font.size = Pt(8.5)
            r.font.color.rgb = RGBColor(15, 41, 66) if is_peak else RGBColor(45, 55, 72)
            if is_peak:
                r.font.bold = True

    # 6. Landasan Literatur
    add_sec_heading("6. Landasan Literatur & Kajian Teori Ilmiah Terkini")
    add_body_p("Untuk memperkokoh dasar metodologi pada laporan akhir, tim telah menelaah dan mengarsipkan 3 publikasi ilmiah utama di direktori papers/ repositori proyek:")
    add_bullet_p(
        "ROLE Algorithm (Deng et al., arXiv:2607.26607, 2026): ",
        "Membuktikan secara matematis bahwa dalam audio open-set, fungsi Softmax terdistorsi oleh keberadaan outlier (softmax confidence trap). Mengusulkan pembobotan latent inlierness pada inferensi transduktif untuk memperbarui prototipe tanpa kontaminasi label asing."
    )
    add_bullet_p(
        "Multi-Label Open-Set Audio Taxonomy (DCASE Workshop 2023, arXiv:2310.13759): ",
        "Memformalkan taksonomi pemisahan suara: Known-Known (KK: spesies target), Known-Unknown (KU: derau hutan latar belakang), dan Unknown-Unknown (UU: spesies liar baru saat uji)."
    )
    add_bullet_p(
        "Energy-Based Out-of-Distribution Detection (Liu et al., NeurIPS 2020, arXiv:2010.03759): ",
        "Membuktikan bahwa Helmholtz Free Energy Score E(x) = -T * log sum(exp(z_c/T)) selaras dengan kerapatan probabilitas fisik sinyal dan jauh lebih tahan banting dibanding probabilitas posterior dalam mendeteksi anomali."
    )

    # 7. Strategi Penilaian Akhir & Rencana Kerja
    add_sec_heading("7. Strategi Penilaian Akhir & Rencana Kerja Tindak Lanjut")
    add_body_p(
        "Sesuai regulasi resmi panduan Halaman 9-10, Nilai Akhir Babak Penyisihan ditentukan oleh formula: "
        "Nilai Akhir = 0.35 x Skor Prediksi Kaggle + 0.65 x Skor Laporan Teknis. "
        "Dengan bobot laporan sebesar 65% (Evaluasi & Validasi 30%, Modeling 25%, Preprocessing 20%, EDA 15%, Struktur 10%), keunggulan metodologi bebas bocor, pengujian hipotesis Kruskal-Wallis, dan forensik taksonomi tim kami menjadi modal kunci untuk menembus 5 Tim Teratas (Top 5) Lolos Babak Final di Kampus ITS Surabaya."
    )
    add_bullet_p(
        "Eksekusi Rencana Day 4 (17 Sel Ramping): ",
        "Menjalankan planning_day_4_experiment.md (In-Fold Pseudo-Labeling 1543-d Hybrid Input + Nested Knee-Point Thresholding) untuk menguji potensi lonjakan skor menuju 0.70+."
    )
    add_bullet_p(
        "Tri-Submission Anti-Shakeup: ",
        "Mengirimkan 3 berkas terukur: sub_014 (Safe Anchor tanpa self-training untuk jaring pengaman 70% Private LB), sub_015 (SOTA Champion Self-Trained 43.5%), dan sub_016 (Conservative Probe 41.5%)."
    )
    add_bullet_p(
        "Penulisan Laporan Teknis 10 Halaman: ",
        "Menyusun dokumen formal DatSci_disuruhryudaftar_Laporan.pdf lengkap dengan visualisasi diagnostik publikasi, tabel ablasi multi-pilar, dan referensi paper ilmiah."
    )

    doc.save(str(output_path))
    print(f"Word document successfully created: {Path(output_path).resolve()} ({Path(output_path).stat().st_size} bytes)")
    return output_path

if __name__ == "__main__":
    create_dosen_report_docx("Laporan_Kemajuan_Schematics_2026_Dosen.docx")

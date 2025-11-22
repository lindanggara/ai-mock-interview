# 🎯 Simulator Interview Data Science

**Platform Latihan Interview Terlengkap untuk Posisi Data Science**

Simulator interview komprehensif yang dirancang khusus untuk posisi Data Science di Indonesia. Latih kemampuan interview Anda dengan pertanyaan real, dapatkan feedback instant, dan track progress Anda - semuanya powered by algoritma text mining advanced.

---

## ✨ Fitur Utama

### 🎯 **Fokus 100% Data Science**
- 10+ kategori mencakup semua topik interview DS
- Pertanyaan sesuai dengan kebutuhan job real
- Level kesulitan: Junior → Mid-level → Senior
- Pertanyaan dalam Bahasa Indonesia yang natural

### 📄 **Analisis CV & Personalisasi**
- Upload CV Anda (PDF/DOCX)
- Ekstraksi skill otomatis
- Deteksi level pengalaman
- Rekomendasi pertanyaan personal
- Identifikasi skill gap

### 🎤 **Dua Mode Interview**
- **💬 Mode Teks**: Ketik jawaban (tradisional)
- **🎙️ Mode Suara**: Bicara jawaban (simulasi realistis)
- Transkripsi otomatis
- Feedback cara penyampaian suara

### 🔬 **Analisis Mendalam**
- 8+ algoritma text mining
- Analisis TF-IDF
- Cosine similarity dengan best practices
- Ekstraksi pola N-gram
- Named entity recognition
- Analisis sentimen & tone
- Scoring readability
- Pengukuran coherence

### 💡 **Sistem Feedback Detail**

Bukan hanya summary generic, tapi feedback lengkap:

1. **Jawaban Anda** - Apa yang Anda tulis
2. **Jawaban Terbaik** - Apa yang diharapkan interviewer
3. **Perbandingan** - Analisis point-by-point
4. **Feedback Spesifik** - Saran actionable
5. **Analisis Gap** - Apa yang terlewat
6. **Ringkasan** - Assessment keseluruhan

### 📊 **Dashboard Analitik**
- Track progress dari waktu ke waktu
- Trend dan pola skor
- Performa per kategori
- Identifikasi kekuatan & kelemahan

### 🎨 **UI/UX Modern**
- Design gradient yang indah
- Animasi smooth
- Navigasi intuitif
- Responsive untuk mobile
- Interface Bahasa Indonesia

---

## 🚀 Cara Install

### Instalasi

```bash
# 1. Clone atau download project
git clone <repo-url>
cd ds-interview-simulator

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"

# 5. Jalankan aplikasi
streamlit run app.py
```

### Setup Pertama Kali

1. **Upload CV Anda** (opsional tapi direkomendasikan)
   - Klik "Upload CV" di sidebar
   - Sistem analisis skill & pengalaman Anda
   - Dapat rekomendasi personal

2. **Pilih Mode Interview**
   - Mode Teks: Ketik jawaban
   - Mode Suara: Bicara jawaban

3. **Pilih Topik & Level**
   - Pilih dari 10 kategori DS
   - Pilih level pengalaman Anda

4. **Latihan & Tingkatkan**
   - Jawab pertanyaan
   - Dapat feedback detail
   - Track progress Anda

---

## 📚 Kategori Pertanyaan

### 1. **Keahlian Teknis & Background**
Tes kemampuan Anda menjelaskan pengalaman teknis dengan contoh konkret.

### 2. **Pengetahuan Statistik**
Jelaskan konsep statistik kompleks dengan bahasa sederhana.

### 3. **Data Wrangling & EDA**
Demonstrasi skill data cleaning dan exploratory analysis.

### 4. **Problem Solving & Studi Kasus**
Walk through pendekatan proyek DS end-to-end.

### 5. **Pemahaman Bisnis**
Hubungkan pekerjaan teknis dengan nilai bisnis.

### 6. **Komunikasi & Kolaborasi**
Jelaskan cara kerja dengan stakeholder non-teknis.

### 7. **Evaluasi & Optimasi Model**
Diskusi strategi evaluasi performa dan tuning model.

### 8. **Big Data & Skalabilitas**
Handle pertanyaan pemrosesan data skala besar.

### 9. **Etika & Bias dalam AI**
Address fairness dan bias dalam ML systems.

### 10. **MLOps & Deployment**
Jelaskan strategi deployment production.

---

## 🎓 Cara Menggunakan Secara Efektif

### Sebelum Latihan

1. **Siapkan CV Anda**
   - Update dengan proyek terbaru
   - Highlight skill relevan
   - Kuantifikasi pencapaian

2. **Upload CV ke Platform**
   - Dapatkan analisis skill
   - Identifikasi gap
   - Lihat tips personal

### Saat Latihan

1. **Gunakan Metode STAR**
   - **S**ituasi: Set konteks
   - **T**ugas: Define tantangan
   - **A**ksi: Jelaskan pendekatan Anda
   - **R**esult: Share hasil (dengan angka!)

2. **Bicara/Tulis Natural**
   - Jangan hafalan
   - Be conversational
   - Gunakan contoh
   - Tunjukkan antusiasme

3. **Review Feedback dengan Teliti**
   - Baca perbandingan
   - Catat point yang terlewat
   - Implementasi saran

### Setelah Tiap Latihan

1. **Analisis Gap**
   - Apa yang terlewat?
   - Kenapa terlewat?
   - Bagaimana improve?

2. **Tulis Ulang Jawaban**
   - Incorporate feedback
   - Tambah elemen yang hilang
   - Bandingkan skor

3. **Track Progress**
   - Cek dashboard analitik
   - Identifikasi pola
   - Fokus pada area lemah

---

## 💡 Contoh: Jawaban Bagus vs Kurang Bagus

### ❌ Jawaban Kurang Bagus (Skor: 2.5/5.0)

> "Saya tahu Python dan pernah pakai untuk analisis data. Saya sudah buat beberapa proyek machine learning dan bisa kerja dengan data. Saya paham pandas dan numpy dan pernah pakai sebelumnya."

**Masalah:**
- Terlalu vague
- Tidak ada contoh spesifik
- Tidak ada hasil terukur
- Tidak ada konteks atau dampak
- Statement generic

### ✅ Jawaban Bagus (Skor: 4.8/5.0)

> "Saya punya pengalaman 3 tahun dengan Python untuk data science. Di proyek terakhir menganalisis churn pelanggan untuk perusahaan e-commerce, saya gunakan pandas untuk manipulasi 2 juta record transaksi dengan 15 fitur. Saya implementasi feature engineering pakai numpy array, buat rolling windows dan agregasi time-based. Untuk modeling, saya pakai RandomForestClassifier dan XGBoost dari scikit-learn, mencapai akurasi 87% dengan F1-score 0.82. Model ini identifikasi 15 ribu pelanggan berisiko, dan kampanye retensi kami selamatkan revenue Rp 7 miliar per tahun. Saya deploy model pakai Flask API dengan Docker, handle 1000+ prediksi per detik."

**Kenapa Bagus:**
- ✅ Tools & library spesifik
- ✅ Angka konkret (2M records, 15 fitur)
- ✅ Detail teknis (RandomForest, XGBoost)
- ✅ Hasil terukur (87% akurasi, Rp 7M saved)
- ✅ Pipeline lengkap (development → deployment)
- ✅ Dampak bisnis jelas

---

## 🎤 Panduan Mode Suara

### Setup

1. **Cek Microphone**
   - Izinkan akses microphone browser
   - Tes level audio
   - Gunakan headset untuk kualitas lebih baik

2. **Environment**
   - Cari lokasi tenang
   - Minimalisir background noise
   - Lighting yang baik

### Best Practices

✅ **Lakukan:**
- Bicara dengan kecepatan normal
- Ucapkan dengan jelas
- Jeda antar pikiran
- Gunakan bahasa natural
- Struktur dengan metode STAR

❌ **Jangan:**
- Terburu-buru
- Terlalu banyak filler words (eee, anu)
- Bicara terlalu pelan
- Ngomong tanpa struktur
- Baca dari catatan

### Tips untuk Istilah Teknis

- Eja akronim: "S-Q-L" bukan "sequel"
- Jeda sebelum istilah teknis
- Gunakan context clues
- Bicara sedikit lambat untuk term kompleks

---

## 📊 Memahami Skor Anda

### Breakdown Skor Keseluruhan

| Skor | Level | Kesiapan Interview |
|------|-------|-------------------|
| 4.5-5.0 | Luar Biasa | Siap untuk interview top companies |
| 4.0-4.4 | Sangat Baik | Siap untuk kebanyakan companies |
| 3.5-3.9 | Bagus | Perlu sedikit polish |
| 3.0-3.4 | Cukup | Latih lagi, fokus contoh |
| 2.5-2.9 | Kurang | Review konsep, tambah detail |
| <2.5 | Perlu Kerja | Pelajari fundamental, terus latihan |

### Komponen Skor

**Akurasi Teknis (40%)**
- Cakupan keyword
- Penyebutan tool/library
- Kedalaman teknis
- Alignment dengan best practices

**Kedalaman Pengetahuan (30%)**
- Kelengkapan jawaban
- Sophistication teknis
- Pendekatan problem-solving
- Inovasi/kreativitas

**Komunikasi Clarity (30%)**
- Struktur kalimat
- Logical flow
- Readability
- Professional tone

---

## 🔧 Kustomisasi

### Menambah Pertanyaan Sendiri

Edit `data/questions.json`:

```json
{
  "Kategori Anda": {
    "question": "Pertanyaan interview Anda di sini",
    "keywords": ["keyword1", "keyword2"],
    "ideal_length": [150, 300],
    "weight": {
      "technical": 0.4,
      "depth": 0.3,
      "structure": 0.3
    }
  }
}
```

### Adjust Bobot Scoring

Modifikasi nilai `weight` di questions.json:
- Higher `technical`: Emphasize tools/keywords
- Higher `depth`: Value kelengkapan/sophistication
- Higher `structure`: Prioritas komunikasi

---

## 🐛 Troubleshooting

### Error: punkt_tab not found

```bash
# Solusi 1: Download resource spesifik
python -c "import nltk; nltk.download('punkt_tab')"

# Solusi 2: Download semua
python -c "import nltk; nltk.download('all')"
```

### CV Upload Tidak Bekerja

```bash
# Install library PDF/DOCX
pip install PyPDF2 python-docx
```

### Issues Mode Suara

Mode suara saat ini menggunakan placeholder transcription. Untuk voice recognition real:

```bash
pip install SpeechRecognition pydub
# Mungkin perlu: pip install pyaudio (requires system libraries)
```

---

## 📈 Rencana Belajar

### Minggu 1-2: Fundamental
- Latihan pertanyaan technical skills
- Fokus pada metode STAR
- Buat template jawaban
- **Target**: Rata-rata 3.5+

### Minggu 3-4: Kedalaman
- Tambah detail teknis lebih
- Kuantifikasi semua hasil
- Latihan ML/Statistics questions
- **Target**: Rata-rata 4.0+

### Minggu 5-6: Polish
- Perfect komunikasi
- Optimasi timing
- Latihan mode suara
- **Target**: Rata-rata 4.5+

### Minggu 7-8: Advanced
- Prep spesifik company
- Mock interviews
- System design questions
- **Target**: Ready for any interview

---

## 💼 Tips Interview Real

### Sebelum Interview

1. **Research Company**
   - Tech stack DS mereka
   - Proyek/blog terbaru
   - Struktur tim

2. **Siapkan Pertanyaan**
   - Tentang tim
   - Tentang proyek
   - Tentang growth

3. **Latihan di Platform**
   - 2-3 pertanyaan daily
   - Track improvement
   - Fokus area lemah

### Saat Interview

1. **Dengarkan Baik-baik**
   - Pahami pertanyaan fully
   - Tanya klarifikasi
   - Pikirkan dulu sebelum bicara

2. **Gunakan Struktur STAR**
   - Selalu ada konteks
   - Explain thinking Anda
   - Show hasil

3. **Be Conversational**
   - Jangan hafalan
   - Tunjukkan antusiasme
   - Jujur tentang limitasi

### Setelah Interview

1. **Catat Pertanyaan yang Ditanya**
   - Latihan lagi
   - Bandingkan dengan database kami
   - Improve jawaban

2. **Self-Assess**
   - Apa yang berjalan baik?
   - Apa yang perlu improve?
   - Topik baru untuk dipelajari?

---

## 🤝 Kontribusi

Ingin improve platform ini?

### Tambah Pertanyaan
Submit pertanyaan baru via pull request ke `data/questions.json`

### Improve Algoritma
Enhance text mining di `src/text_mining.py`

### Better UI/UX
Improve design di section CSS `app.py`

### Dokumentasi
Bantu yang lain dengan panduan dan tutorial

---

## 📄 Lisensi

MIT License - Bebas digunakan dan dimodifikasi

---

## 🙏 Acknowledgments

- Dibangun dengan Streamlit
- NLP powered by NLTK & TextBlob
- ML algorithms dari scikit-learn
- Visualisasi by Plotly

---

**Mulai Latihan Hari Ini!** 🚀

```bash
streamlit run app.py
```

Dream job Data Science Anda tinggal beberapa latihan lagi! 💪

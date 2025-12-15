# 🚀 INTRVU

Sistem wawancara kerja *hybrid* yang menggabungkan konferensi video real-time dengan analisis pasca-wawancara menggunakan Kecerdasan Buatan (AI). Sistem ini membantu HRD menilai kandidat secara objektif berdasarkan relevansi jawaban (Semantik), ketepatan teknis (Keyword), dan gaya komunikasi (WPM & Filler Words).

## 🌟 Fitur Utama

### 👨‍💼 HRD Dashboard (`hrd.py`)
* **Live Video Conference:** Terintegrasi dengan Jitsi Meet untuk wawancara tatap muka.
* **AI Transcription:** Mengubah suara menjadi teks otomatis menggunakan **OpenAI Whisper**.
* **Semantic Scoring:** Menilai kesesuaian makna jawaban kandidat dengan kunci jawaban menggunakan **Sentence Transformers** (`all-MiniLM-L6-v2`).
* **Keyword Analysis:** Mendeteksi apakah kandidat menyebutkan istilah teknis wajib.
* **Communication Metrics:** Mengukur *Words Per Minute* (WPM) dan mendeteksi *Filler Words* (seperti "ehm", "anu").
* **Automated Reporting:** Menghasilkan laporan analisis dalam bentuk **PDF** dan grafik Radar Chart.
* **Multilingual Support:** Mendukung analisis video Bahasa Indonesia & Inggris (dilengkapi Auto-Translation).

### 👤 Candidate Portal (`candidate_app.py`)
* **Easy Access:** Kandidat cukup memasukkan Nama dan Kode Meeting.
* **Secure Room:** Room video dibuat dinamis berdasarkan kode unik dari HRD.
* **User Friendly:** Antarmuka sederhana tanpa perlu login akun rumit.

---

## 🛠️ Teknologi yang Digunakan

* **Framework:** [Streamlit](https://streamlit.io/)
* **Speech-to-Text:** [OpenAI Whisper](https://github.com/openai/whisper)
* **NLP & Embedding:** [Sentence-Transformers](https://www.sbert.net/) (HuggingFace)
* **Video Conference:** [Jitsi Meet](https://meet.jit.si/) (Iframe Integration)
* **Visualization:** Plotly (Radar Chart)
* **Reporting:** FPDF (PDF Generation)

---

## LANGKAH - LANGKAH

### HALAMAN HRD
1. Buka halaman web milik HRD
   * https://chisa06-intrv-hrd.hf.space
2. Konfigurasi parameter
   * Pilih Bahasa Video (Indonesia/Inggris).
   * Masukkan Jawaban Ideal (Kunci jawaban untuk referensi AI).
   * Masukkan Keywords Wajib (Istilah teknis yang harus muncul).
3. Buat sesi live
   * Klik "Generate Kode".
   * Bagikan kode tersebut ke kandidat.
   * Tombol Record di aplikasi perekam layar Anda (OBS/Game Bar) secara manual, karena Jitsi Embed tidak merekam otomatis.
4. Analisis video
   * Upload file video hasil rekaman (.mp4, .webm, .wav).
   * Klik "Mulai Analisis AI".
   * Unduh laporan PDF setelah analisis selesai.

---

### HALAMAN KANDIDAT
1. Buka halaman web milik kandidat
   * https://chisa06-intrvu.hf.space
2. Login ke room meet
   * Masukkan Nama Lengkap.
   * Masukkan Kode Meeting yang diberikan HRD.
   * Klik "Masuk ke Room 🚀".

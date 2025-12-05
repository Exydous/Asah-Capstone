import streamlit as st
import whisper
import pandas as pd
import os
import tempfile
from sentence_transformers import SentenceTransformer, util

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Interview Assessor", layout="wide")

st.title("🤖 AI Interview Assessor")
st.markdown("""
Aplikasi ini menilai video wawancara secara otomatis menggunakan **OpenAI Whisper** (Speech-to-Text) 
dan **Sentence-BERT** (Semantic Analysis).
""")

# --- CACHE MODELS (Agar tidak loading ulang setiap klik) ---
@st.cache_resource
def load_models():
    print("Loading Models...")
    stt_model = whisper.load_model("base")
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Models Loaded!")
    return stt_model, nlp_model

# Load models di awal
stt_model, nlp_model = load_models()

# --- HELPER FUNCTIONS (Dari Notebook Anda) ---
def analisis_text_detail(transkrip, keywords_wajib):
    transkrip_lower = transkrip.lower()
    
    # 1. Cek Filler Words
    filler_list = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'i mean']
    filler_count = 0
    found_fillers = []
    
    for filler in filler_list:
        # Tambah spasi agar tidak mendeteksi kata di dalam kata lain (misal: 'like' di 'likely')
        count = transkrip_lower.count(f" {filler} ") 
        if count > 0:
            filler_count += count
            found_fillers.append(f"{filler}({count})")
            
    # 2. Cek Keywords
    missed_keywords = []
    hit_keywords = []
    
    for key in keywords_wajib:
        if key.lower() in transkrip_lower:
            hit_keywords.append(key)
        else:
            missed_keywords.append(key)
            
    if len(keywords_wajib) == 0:
        keyword_score = 100
    else:
        keyword_score = (len(hit_keywords) / len(keywords_wajib)) * 100
        
    return {
        'filler_count': filler_count,
        'filler_details': ", ".join(found_fillers),
        'missed_keywords': missed_keywords,
        'keyword_score': keyword_score
    }

def generate_automated_feedback(wpm, semantic_score, missed_keywords, filler_count):
    feedback = []
    
    # WPM
    if wpm < 110:
        feedback.append("⚠️ Bicara terlalu lambat, coba tingkatkan energi.")
    elif wpm > 160:
        feedback.append("⚠️ Bicara terlalu cepat, pelankan sedikit agar jelas.")
    else:
        feedback.append("✅ Kecepatan bicara (pacing) sudah bagus.")
        
    # Keywords
    if len(missed_keywords) > 0:
        feedback.append(f"❌ Kamu lupa menyebutkan konsep: '{', '.join(missed_keywords)}'.")
    else:
        feedback.append("✅ Semua poin kunci tersampaikan.")
        
    # Fillers
    if filler_count > 4:
        feedback.append("⚠️ Kurangi kata 'um/uh/like' agar lebih profesional.")
        
    # Semantic
    if semantic_score < 60:
        feedback.append("⚠️ Jawaban kurang relevan dengan definisi ideal.")
        
    return " ".join(feedback)

# --- SIDEBAR: INPUT DATA ---
with st.sidebar:
    st.header("1. Upload Database Jawaban")
    file_kunci = st.file_uploader("Upload Excel Kunci Jawaban (.xlsx)", type=['xlsx'])
    
    st.header("2. Upload Video Kandidat")
    uploaded_videos = st.file_uploader("Pilih Video Interview", type=['webm', 'mp4', 'wav'], accept_multiple_files=True)
    
    analyze_btn = st.button("Mulai Analisis 🚀")

# --- PROSES UTAMA ---
if analyze_btn:
    if not file_kunci or not uploaded_videos:
        st.error("Mohon upload file Excel Kunci Jawaban DAN Video terlebih dahulu!")
    else:
        # 1. Load Kunci Jawaban
        try:
            df_kunci = pd.read_excel(file_kunci)
            # Konversi ke Dictionary agar mudah diakses
            kunci_jawaban_db = {}
            for _, row in df_kunci.iterrows():
                fname = str(row['filename_id']).strip()
                kwd_raw = str(row['keywords'])
                kwd_list = [k.strip() for k in kwd_raw.split(',') if k.strip()]
                
                kunci_jawaban_db[fname] = {
                    'ideal_answer': str(row['ideal_answer']),
                    'keywords': kwd_list
                }
            st.success(f"✅ Database dimuat: {len(kunci_jawaban_db)} soal ditemukan.")
        except Exception as e:
            st.error(f"Gagal membaca Excel: {e}")
            st.stop()

        # 2. Proses Setiap Video
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, video_file in enumerate(uploaded_videos):
            filename_clean = os.path.splitext(video_file.name)[0]
            status_text.text(f"Sedang memproses: {video_file.name}...")
            
            # Cek apakah ada kuncinya
            if filename_clean not in kunci_jawaban_db:
                st.warning(f"Skipping {video_file.name}: Tidak ada kunci jawaban untuk ID '{filename_clean}'")
                continue
                
            # Simpan file sementara agar bisa dibaca Whisper
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_file:
                tmp_file.write(video_file.read())
                tmp_path = tmp_file.name
            
            try:
                # A. Transcribe
                result = stt_model.transcribe(tmp_path, language='en')
                transkrip = result['text'].strip()
                
                # B. Hitung Metrik
                durasi = result['segments'][-1]['end'] if result['segments'] else 1.0
                jumlah_kata = len(transkrip.split())
                wpm = (jumlah_kata / durasi) * 60
                
                # C. Ambil Kunci & Bandingkan
                data_kunci = kunci_jawaban_db[filename_clean]
                
                # Semantic Score
                emb_kandidat = nlp_model.encode(transkrip, convert_to_tensor=True)
                emb_ideal = nlp_model.encode(data_kunci['ideal_answer'], convert_to_tensor=True)
                semantic_score = util.pytorch_cos_sim(emb_kandidat, emb_ideal).item() * 100
                
                # Keyword Analysis
                analisis = analisis_text_detail(transkrip, data_kunci['keywords'])
                
                # Final Score
                final_score = (semantic_score * 0.6) + (analisis['keyword_score'] * 0.4)
                
                # Feedback
                feedback = generate_automated_feedback(
                    wpm, semantic_score, analisis['missed_keywords'], analisis['filler_count']
                )
                
                results.append({
                    "Video File": video_file.name,
                    "Final Score": round(final_score, 1),
                    "WPM": int(wpm),
                    "Fillers": analisis['filler_count'],
                    "Transcript": transkrip,
                    "Missed Keywords": ", ".join(analisis['missed_keywords']),
                    "Feedback": feedback
                })
                
            except Exception as e:
                st.error(f"Error pada file {video_file.name}: {e}")
            finally:
                # Hapus file temporary
                os.remove(tmp_path)
            
            # Update Progress
            progress_bar.progress((idx + 1) / len(uploaded_videos))
            
        status_text.text("Analisis Selesai!")
        
        # 3. Tampilkan Hasil
        if results:
            df_results = pd.DataFrame(results)
            
            st.divider()
            st.subheader("📊 Rangkuman Penilaian")
            
            # Tampilkan Tabel Utama
            st.dataframe(df_results[['Video File', 'Final Score', 'WPM', 'Fillers', 'Feedback']].style.background_gradient(subset=['Final Score'], cmap='RdYlGn'), use_container_width=True)
            
            st.subheader("📝 Detail Analisis per Kandidat")
            for index, row in df_results.iterrows():
                with st.expander(f"Detail: {row['Video File']} (Skor: {row['Final Score']})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Transkrip:**")
                        st.info(row['Transcript'])
                        st.markdown("**Saran Perbaikan:**")
                        st.warning(row['Feedback'])
                        
                    with col2:
                        st.metric("Words Per Minute", f"{row['WPM']} wpm")
                        st.metric("Filler Words", f"{row['Fillers']} kali")
                        if row['Missed Keywords']:
                            st.error(f"**Missed:** {row['Missed Keywords']}")
                        else:
                            st.success("**Keywords Lengkap!**")

else:
    st.info("👈 Silakan upload file Kunci Jawaban (Excel) dan Video di menu sebelah kiri untuk memulai.")
    
    # Template Download Helper
    st.markdown("---")
    st.markdown("### Belum punya template Excel?")
    
    # Membuat tombol download template dummy
    dummy_data = {
        'filename_id': ['interview_question_1', 'interview_question_2'],
        'ideal_answer': [
            'Machine Learning is a subset of AI that focuses on building systems that learn from data.',
            'Supervised learning uses labeled data while unsupervised uses unlabeled data.'
        ],
        'keywords': ['learn from data, systems', 'labeled data, unlabeled']
    }
    df_dummy = pd.DataFrame(dummy_data)
    
    # Convert ke Excel di memory
    from io import BytesIO
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_dummy.to_excel(writer, index=False)
        
    st.download_button(
        label="Download Template Excel Kunci Jawaban",
        data=buffer.getvalue(),
        file_name="template_kunci_jawaban.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
import streamlit as st
import whisper
import pandas as pd
import os
import tempfile
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Interview Assessor + Translator", layout="wide")

st.title("🤖 AI Interview Assessor (with Translation)")
st.markdown("""
Aplikasi ini menilai video wawancara dan **menerjemahkan jawaban kandidat ke Bahasa Indonesia** menggunakan model Neural Machine Translation.
""")

# --- CACHE MODELS ---
@st.cache_resource
def load_models():
    print("Loading Models...")
    # 1. Model Speech-to-Text (Whisper)
    stt_model = whisper.load_model("base")
    
    # 2. Model Semantic Similarity
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Model Translation (English -> Indonesian)
    # Menggunakan model MarianMT dari Helsinki-NLP yang efisien
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")
    
    print("All Models Loaded!")
    return stt_model, nlp_model, translator

# Load semua model
stt_model, nlp_model, translator = load_models()

# --- HELPER FUNCTIONS ---
def translate_long_text(text, translator_pipeline):
    """
    Memecah teks panjang menjadi potongan-potongan kecil agar muat di model translasi
    (Model biasanya punya limit 512 token).
    """
    if not text:
        return ""
        
    # Pecah berdasarkan kalimat (titik)
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Jika chunk sudah cukup panjang, simpan dan buat chunk baru
        if len(current_chunk) + len(sentence) < 400: # Estimasi aman karakter
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    chunks.append(current_chunk) # Append sisa terakhir
    
    # Translate per chunk
    translated_chunks = []
    for chunk in chunks:
        if chunk.strip():
            # truncation=True untuk berjaga-jaga
            res = translator_pipeline(chunk, max_length=512, truncation=True)
            translated_chunks.append(res[0]['translation_text'])
            
    return " ".join(translated_chunks)

def analisis_text_detail(transkrip):
    transkrip_lower = transkrip.lower()
    
    # Cek Filler Words
    filler_list = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'i mean', 'eum', 'anu']
    filler_count = 0
    found_fillers = []
    
    for filler in filler_list:
        count = transkrip_lower.count(f" {filler} ") 
        if count > 0:
            filler_count += count
            found_fillers.append(f"{filler}({count})")
            
    return {
        'filler_count': filler_count,
        'filler_details': ", ".join(found_fillers)
    }

def generate_automated_feedback(wpm, semantic_score, filler_count):
    feedback = []
    
    if wpm < 110:
        feedback.append("⚠️ Bicara terlalu lambat.")
    elif wpm > 160:
        feedback.append("⚠️ Bicara terlalu cepat.")
    else:
        feedback.append("✅ Pacing bagus.")
        
    if filler_count > 4:
        feedback.append("⚠️ Kurangi filler words.")
        
    if semantic_score < 60:
        feedback.append("⚠️ Jawaban kurang relevan.")
    else:
        feedback.append("✅ Jawaban relevan.")
        
    return " ".join(feedback)

# --- SIDEBAR: INPUT DATA ---
with st.sidebar:
    st.header("1. Masukkan Kunci Jawaban")
    st.info("Masukkan jawaban ideal dalam Bahasa Inggris (jika video Inggris) atau Indonesia.")
    
    input_ideal_answer = st.text_area(
        "Jawaban Ideal (Reference)", 
        height=150,
        placeholder="Contoh: Machine Learning is a branch of AI..."
    )
    
    st.divider()
    
    st.header("2. Upload Video Kandidat")
    uploaded_videos = st.file_uploader(
        "Pilih Video Interview", 
        type=['webm', 'mp4', 'wav', 'mkv', 'mov'], 
        accept_multiple_files=True
    )
    
    analyze_btn = st.button("Mulai Analisis 🚀", type="primary")

# --- PROSES UTAMA ---
if analyze_btn:
    if not input_ideal_answer:
        st.error("⚠️ Mohon isi 'Jawaban Ideal' terlebih dahulu!")
    elif not uploaded_videos:
        st.error("⚠️ Mohon upload video!")
    else:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, video_file in enumerate(uploaded_videos):
            status_text.text(f"Sedang memproses: {video_file.name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp_file:
                tmp_file.write(video_file.read())
                tmp_path = tmp_file.name
            
            try:
                # A. Transcribe (Whisper)
                # Note: Whisper akan mendeteksi bahasa asli video (misal: Inggris)
                result = stt_model.transcribe(tmp_path) 
                transkrip_asli = result['text'].strip()
                
                # B. Machine Learning Translation (En -> Id)
                status_text.text(f"Menerjemahkan: {video_file.name}...")
                transkrip_indo = translate_long_text(transkrip_asli, translator)
                
                # C. Hitung Metrik
                durasi = result['segments'][-1]['end'] if result['segments'] else 1.0
                jumlah_kata = len(transkrip_asli.split())
                wpm = (jumlah_kata / durasi) * 60
                
                # D. Analisis Semantik (Menggunakan Transkrip Asli vs Ideal Answer Asli)
                # Asumsi: Jawaban Ideal yang diinput user bahasanya SAMA dengan bahasa video kandidat
                emb_kandidat = nlp_model.encode(transkrip_asli, convert_to_tensor=True)
                emb_ideal = nlp_model.encode(input_ideal_answer, convert_to_tensor=True)
                
                semantic_score = util.pytorch_cos_sim(emb_kandidat, emb_ideal).item() * 100
                analisis = analisis_text_detail(transkrip_asli)
                
                final_score = semantic_score 
                
                feedback = generate_automated_feedback(
                    wpm, semantic_score, analisis['filler_count']
                )
                
                results.append({
                    "Video File": video_file.name,
                    "Final Score": round(final_score, 1),
                    "WPM": int(wpm),
                    "Transcript (Original)": transkrip_asli,
                    "Transcript (Indo)": transkrip_indo, # Simpan hasil translate
                    "Feedback": feedback
                })
                
            except Exception as e:
                st.error(f"Error pada file {video_file.name}: {e}")
            finally:
                os.remove(tmp_path)
            
            progress_bar.progress((idx + 1) / len(uploaded_videos))
            
        status_text.text("Selesai!")
        
        if results:
            df_results = pd.DataFrame(results)
            
            st.divider()
            st.subheader("📊 Hasil Penilaian & Translasi")

            # Tabel Ringkasan
            st.dataframe(
                df_results[['Video File', 'Final Score', 'WPM', 'Feedback']]
                .style.background_gradient(subset=['Final Score'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Detail per Kandidat
            st.subheader("📝 Detail Transkrip")
            for index, row in df_results.iterrows():
                with st.expander(f"{row['Video File']} - Skor: {row['Final Score']}"):
                    
                    col_orig, col_trans = st.columns(2)
                    
                    with col_orig:
                        st.markdown("**🗣️ Asli (English/Source):**")
                        st.info(row['Transcript (Original)'])
                        
                    with col_trans:
                        st.markdown("**🇮🇩 Terjemahan (Indonesian):**")
                        # Highlight bahwa ini hasil ML
                        st.success(row['Transcript (Indo)'])
                        st.caption("*Diterjemahkan oleh model Helsinki-NLP/opus-mt-en-id*")
                    
                    st.divider()
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Evaluasi AI:**")
                        st.write(row['Feedback'])
                    with c2:
                        st.markdown("**Kunci Jawaban (Ref):**")
                        st.caption(input_ideal_answer[:150] + "...")

else:
    st.info("👈 Masukkan Jawaban Ideal dan Upload Video di menu sebelah kiri.")
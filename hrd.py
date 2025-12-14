import streamlit as st
import streamlit.components.v1 as components
import whisper
import pandas as pd
import os
import tempfile
import random
import string
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="HRD Dashboard - AI Interview", layout="wide")

# --- SESSION STATE ---
if 'meeting_code' not in st.session_state:
    st.session_state['meeting_code'] = None

# --- CACHE MODELS ---
@st.cache_resource
def load_models():
    print("Loading AI Models...")
    stt_model = whisper.load_model("base")
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")
    print("All Models Loaded!")
    return stt_model, nlp_model, translator

stt_model, nlp_model, translator = load_models()

# --- HELPER FUNCTIONS ---
def generate_code():
    # Generate 5 karakter acak (A-Z, 0-9)
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(5))

def translate_long_text(text, translator_pipeline):
    if not text: return ""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 400: 
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    chunks.append(current_chunk) 
    translated_chunks = []
    for chunk in chunks:
        if chunk.strip():
            try:
                res = translator_pipeline(chunk, max_length=512, truncation=True)
                translated_chunks.append(res[0]['translation_text'])
            except:
                translated_chunks.append(chunk)
    return " ".join(translated_chunks)

def analisis_text_detail(transkrip, keywords_wajib):
    transkrip_lower = transkrip.lower()
    filler_list = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'i mean']
    filler_count = sum(transkrip_lower.count(f" {f} ") for f in filler_list)
    
    missed_keywords = []
    hit_keywords = []
    if not keywords_wajib:
        keyword_score = 100
    else:
        for key in keywords_wajib:
            if key.lower() in transkrip_lower:
                hit_keywords.append(key)
            else:
                missed_keywords.append(key)
        keyword_score = (len(hit_keywords) / len(keywords_wajib)) * 100
            
    return {'filler_count': filler_count, 'missed_keywords': missed_keywords, 'keyword_score': keyword_score}

def generate_automated_feedback(wpm, semantic_score, missed_keywords, filler_count):
    feedback = []
    if wpm < 110: feedback.append("⚠️ Bicara terlalu lambat.")
    elif wpm > 160: feedback.append("⚠️ Bicara terlalu cepat.")
    else: feedback.append("✅ Pacing bagus.")
    
    if len(missed_keywords) > 0: feedback.append(f"❌ Poin hilang: '{', '.join(missed_keywords)}'.")
    else: feedback.append("✅ Keyword lengkap.")
        
    if filler_count > 4: feedback.append("⚠️ Kurangi filler words.")
    if semantic_score < 60: feedback.append("⚠️ Jawaban kurang relevan.")
    else: feedback.append("✅ Konsep jawaban relevan.")
    return " ".join(feedback)

# ==========================================
# UI INTERFACE HRD
# ==========================================
st.title("👨‍💼 HRD Command Center")

with st.sidebar:
    st.header("1. Konfigurasi Jawaban")
    bahasa_video = st.radio("Bahasa Video:", ("Inggris", "Indonesia"), index=0)
    input_ideal_answer = st.text_area("Jawaban Ideal", height=150, placeholder="Contoh: Machine Learning is...")
    input_keywords = st.text_input("Keywords Wajib", placeholder="AI, Data, System")
    st.divider()
    st.info("Tab 1: Video Call (Wajib Rekam Layar). Tab 2: Upload & Analisis AI.")

# --- TABS UTAMA ---
tab_meet, tab_analysis = st.tabs(["🎥 Live Meeting Room", "📊 AI Analysis Result"])

# --- TAB 1: LIVE MEETING ROOM (JITSI) ---
with tab_meet:
    col_ctrl, col_video = st.columns([1, 3])
    
    with col_ctrl:
        st.subheader("🔑 Buat Sesi")
        if st.session_state['meeting_code'] is None:
            if st.button("Generate Kode"):
                st.session_state['meeting_code'] = generate_code()
                st.rerun()
        else:
            st.success(f"Kode: **{st.session_state['meeting_code']}**")
            st.caption("Berikan kode ini ke kandidat.")
            
            st.warning("🔴 **PERINGATAN:**")
            st.caption("Sistem ini menggunakan server Jitsi untuk video call. Untuk analisis AI, Anda **WAJIB MEREKAM LAYAR (Screen Record)** sesi ini secara manual, lalu upload ke Tab sebelah.")
            
            if st.button("Akhiri Sesi"):
                st.session_state['meeting_code'] = None
                st.rerun()

    with col_video:
        if st.session_state['meeting_code']:
            # PREFIX UNIK agar room tidak tabrakan dengan orang lain di server Jitsi
            room_name = f"InterviewSystem-{st.session_state['meeting_code']}"
            jitsi_url = (
                    f"https://meet.jit.si/{room_name}"
                    f"#userInfo.displayName=\"HRD Manager\""
                    f"&config.prejoinPageEnabled=false" 
                    f"&config.disableDeepLinking=true"
                )
            
            st.markdown(f"### 🔴 Live: {room_name}")
            components.iframe(jitsi_url, height=600, scrolling=False)
        else:
            st.info("👈 Klik tombol 'Generate Kode' untuk memulai meeting.")

# --- TAB 2: AI ANALYSIS ---
with tab_analysis:
    st.subheader("🕵️‍♀️ Analisis Rekaman Interview")
    uploaded_videos = st.file_uploader("Upload Hasil Rekaman Layar", type=['webm', 'mp4', 'wav', 'mkv'], accept_multiple_files=True)
    analyze_btn = st.button("Mulai Analisis AI 🚀", type="primary")

    if analyze_btn:
        if not input_ideal_answer or not uploaded_videos:
            st.error("Mohon lengkapi Jawaban Ideal dan Upload Video.")
        else:
            keywords_list = [k.strip() for k in input_keywords.split(',') if k.strip()]
            lang_code = "en" if bahasa_video == "Inggris" else "id"
            results = []
            progress_bar = st.progress(0)
            
            for idx, video_file in enumerate(uploaded_videos):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1]) as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name
                try:
                    # 1. Transkripsi
                    result = stt_model.transcribe(tmp_path, language=lang_code)
                    transkrip_asli = result['text'].strip()
                    
                    # 2. Translasi
                    if lang_code == 'en': transkrip_indo = translate_long_text(transkrip_asli, translator)
                    else: transkrip_indo = transkrip_asli
                    
                    # 3. Analisis
                    durasi = result['segments'][-1]['end'] if result['segments'] else 1.0
                    wpm = (len(transkrip_asli.split()) / durasi) * 60
                    
                    emb1 = nlp_model.encode(transkrip_asli, convert_to_tensor=True)
                    emb2 = nlp_model.encode(input_ideal_answer, convert_to_tensor=True)
                    sem_score = util.pytorch_cos_sim(emb1, emb2).item() * 100
                    
                    analisis = analisis_text_detail(transkrip_asli, keywords_list)
                    
                    if keywords_list:
                        final_score = (sem_score * 0.6) + (analisis['keyword_score'] * 0.4)
                    else:
                        final_score = sem_score
                        
                    feedback = generate_automated_feedback(wpm, sem_score, analisis['missed_keywords'], analisis['filler_count'])
                    
                    results.append({
                        "Video File": video_file.name,
                        "Final Score": round(final_score, 1),
                        "WPM": int(wpm),
                        "Transcript": transkrip_asli,
                        "Translate": transkrip_indo,
                        "Feedback": feedback
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.remove(tmp_path)
                progress_bar.progress((idx + 1) / len(uploaded_videos))
                
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df[['Video File', 'Final Score', 'WPM', 'Feedback']])
                for i, r in df.iterrows():
                    with st.expander(f"Detail Transkrip: {r['Video File']}"):
                        st.write(f"**Feedback:** {r['Feedback']}")
                        c1, c2 = st.columns(2)
                        c1.info(f"**Asli:**\n{r['Transcript']}")
                        c2.success(f"**Indo:**\n{r['Translate']}")


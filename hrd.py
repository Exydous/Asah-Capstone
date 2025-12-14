import streamlit as st
import streamlit.components.v1 as components
import whisper
import pandas as pd
import os
import tempfile
import random
import string
import re
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import plotly.express as px
from fpdf import FPDF

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

def create_radar_chart(semantic, keyword, wpm, filler):
    # Normalisasi skor komunikasi (Kasarannya)
    # WPM ideal 110-160. Jika di luar range, skor turun.
    if 110 <= wpm <= 160: comm_score = 100
    else: comm_score = max(0, 100 - abs(135 - wpm)) # Penalti jarak dari rata-rata
    
    # Penalti filler words
    comm_score = max(0, comm_score - (filler * 5))

    df = pd.DataFrame(dict(
        r=[semantic, keyword, comm_score],
        theta=['Relevansi (Semantic)', 'Teknis (Keyword)', 'Komunikasi (Flow)']
    ))
    
    fig = px.line_polar(df, r='r', theta='theta', line_close=True, range_r=[0,100])
    fig.update_traces(fill='toself', line_color='#4682B4')
    fig.update_layout(margin=dict(t=20, b=20, l=40, r=40))
    return fig

def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Laporan Interview: {data['Video File']}", ln=1, align='C')
    pdf.ln(10)
    
    # Scores
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Final Score: {data['Final Score']}/100", ln=1)
    pdf.cell(200, 10, txt=f"Semantic Match: {data['Semantic Match']}%", ln=1)
    pdf.cell(200, 10, txt=f"Keyword Match: {data['Keyword Match']}%", ln=1)
    pdf.cell(200, 10, txt=f"WPM: {data['WPM']} | Filler Words: {len(data['Feedback'].split('Filler')) if 'Filler' in data['Feedback'] else 0}", ln=1)
    pdf.ln(10)
    
    # Feedback
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Feedback Sistem:", ln=1)
    pdf.set_font("Arial", size=12)
    # Hapus emoji karena FPDF standar tidak support emoji
    clean_feedback = data['Feedback'].encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean_feedback)
    pdf.ln(5)

    # Transcript
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Transkrip Kandidat:", ln=1)
    pdf.set_font("Arial", size=10)
    # Handling text encoding sederhana
    clean_transkrip = data['Transcript'].encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, txt=clean_transkrip)
    
    return pdf.output(dest='S').encode('latin-1')

def generate_code():
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
    """
    Analisis keyword hit/miss dan filler words.
    """
    transkrip_lower = transkrip.lower()
    filler_list = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'i mean', 'anu', 'ehm'] 
    filler_count = sum(transkrip_lower.count(f" {f} ") for f in filler_list)
    
    missed_keywords = []
    hit_keywords = []
    
    if not keywords_wajib:
        keyword_score = 100
    else:
        for key in keywords_wajib:
            if re.search(r"\b" + re.escape(key.lower()) + r"\b", transkrip_lower):
                hit_keywords.append(key)
            elif key.lower() in transkrip_lower: 
                hit_keywords.append(key)
            else:
                missed_keywords.append(key)
        
        keyword_score = (len(hit_keywords) / len(keywords_wajib)) * 100 if keywords_wajib else 100
            
    return {
        'filler_count': filler_count, 
        'missed_keywords': missed_keywords, 
        'hit_keywords': hit_keywords,
        'keyword_score': keyword_score
    }

def highlight_text(text, keywords):
    """
    Mewarnai keyword dengan warna biru lembut (:blue).
    """
    processed_text = text
    for key in keywords:
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        processed_text = pattern.sub(lambda m: f"**:blue[{m.group(0)}]**", processed_text)
    return processed_text

def generate_automated_feedback(wpm, semantic_score, missed_keywords, filler_count):
    feedback = []
    if wpm < 110: feedback.append("⚠️ Bicara terlalu lambat.")
    elif wpm > 160: feedback.append("⚠️ Bicara terlalu cepat.")
    else: feedback.append("✅ Pacing bagus.")
    
    if len(missed_keywords) > 0: feedback.append(f"❌ Poin hilang: {len(missed_keywords)} keyword.")
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
    st.header("Konfigurasi Jawaban")
    bahasa_video = st.radio("Bahasa Video:", ("Inggris", "Indonesia"), index=0)
    input_ideal_answer = st.text_area("Jawaban Ideal", height=150, placeholder="Contoh: Machine Learning is a field of study...")
    input_keywords = st.text_input("Keywords Wajib", placeholder="AI, Data, System")
    st.divider()
    #st.info("Tab 1: Video Call (Wajib Rekam Layar). Tab 2: Upload & Analisis AI.")

# --- TABS UTAMA ---
tab_meet, tab_analysis = st.tabs(["🎥 Live Meeting Room", "📊 AI Analysis Result"])

# --- TAB 1: LIVE MEETING ROOM ---
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
            st.warning("🔴 **PERINGATAN:** Wajib Rekam Layar manual.")
            if st.button("Akhiri Sesi"):
                st.session_state['meeting_code'] = None
                st.rerun()

    with col_video:
        if st.session_state['meeting_code']:
            room_name = f"InterviewSystem-{st.session_state['meeting_code']}"
            jitsi_url = f"https://meet.jit.si/{room_name}#userInfo.displayName=\"HRD Manager\"&config.prejoinPageEnabled=false&config.disableDeepLinking=true"
            st.markdown(f"### 🔴 Live: {room_name}")
            components.iframe(jitsi_url, height=600, scrolling=False)
        #else:
            #st.info("👈 Klik tombol 'Generate Kode' untuk memulai meeting.")

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
                    # Proses AI
                    result = stt_model.transcribe(tmp_path, language=lang_code)
                    transkrip_asli = result['text'].strip()
                    if lang_code == 'en': transkrip_indo = translate_long_text(transkrip_asli, translator)
                    else: transkrip_indo = transkrip_asli
                    
                    durasi = result['segments'][-1]['end'] if result['segments'] else 1.0
                    wpm = (len(transkrip_asli.split()) / durasi) * 60
                    
                    emb1 = nlp_model.encode(transkrip_asli, convert_to_tensor=True)
                    emb2 = nlp_model.encode(input_ideal_answer, convert_to_tensor=True)
                    sem_score = util.pytorch_cos_sim(emb1, emb2).item() * 100
                    
                    analisis = analisis_text_detail(transkrip_asli, keywords_list)
                    final_score = (sem_score * 0.6) + (analisis['keyword_score'] * 0.4) if keywords_list else sem_score
                    feedback = generate_automated_feedback(wpm, sem_score, analisis['missed_keywords'], analisis['filler_count'])
                    
                    results.append({
                        "Video File": video_file.name,
                        "Final Score": round(final_score, 1),
                        "Semantic Match": round(sem_score, 1),
                        "Keyword Match": round(analisis['keyword_score'], 1),
                        "WPM": int(wpm),
                        "Feedback": feedback, # Hapus kolom transcript dari tabel utama agar rapi
                        "Transcript": transkrip_asli,
                        "Translate": transkrip_indo,
                        "Hit Keywords": analisis['hit_keywords'],
                        "Missed Keywords": analisis['missed_keywords']
                    })
                except Exception as e:
                    st.error(f"Error pada file {video_file.name}: {e}")
                finally:
                    if os.path.exists(tmp_path): os.remove(tmp_path)
                progress_bar.progress((idx + 1) / len(uploaded_videos))
                
            if results:
                df = pd.DataFrame(results)
                
                # --- TAMPILAN DASHBOARD UTAMA ---
                st.markdown("### 📋 Ringkasan Hasil")
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                best_candidate = df.loc[df['Final Score'].idxmax()]
                
                with col_metric1:
                    st.metric("Kandidat Terbaik", best_candidate['Video File'])
                with col_metric2:
                    st.metric("Top Score", f"{best_candidate['Final Score']}")
                with col_metric3:
                    st.metric("Rata-rata WPM", f"{int(df['WPM'].mean())}")

                st.dataframe(
                    df[['Video File', 'Final Score', 'Semantic Match', 'Keyword Match', 'WPM', 'Feedback']]
                    .style.highlight_max(axis=0, color='#4682B4')
                )
                
                st.markdown("### 🧐 Detail Analisis & Report")
                
                for i, r in df.iterrows():
                    with st.expander(f"📄 {r['Video File']} (Skor: {r['Final Score']})"):
                        
                        # LAYOUT BARU: Kiri (Grafik) - Kanan (Transkrip)
                        c_chart, c_text = st.columns([1, 2])
                        
                        with c_chart:
                            st.markdown("#### 🕸️ Kompetensi")
                            # 1. Panggil Fungsi Radar Chart
                            fig = create_radar_chart(r['Semantic Match'], r['Keyword Match'], r['WPM'], r['Feedback'].count('Filler'))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("#### 📥 Download Report")
                            # 2. Panggil Fungsi PDF Export
                            pdf_bytes = create_pdf(r)
                            st.download_button(
                                label="Download PDF Report 📑",
                                data=pdf_bytes,
                                file_name=f"Report_{r['Video File']}.pdf",
                                mime="application/pdf",
                                key=f"btn_{i}"
                            )

                        with c_text:
                            st.markdown("#### 🆚 Analisis Jawaban")
                            # Highlight Text Function (Pastikan fungsi highlight_text kamu ada)
                            highlighted_transcript = highlight_text(r['Transcript'], r['Hit Keywords'])
                            
                            st.caption("Jawaban Kandidat:")
                            with st.container(border=True):
                                st.markdown(highlighted_transcript)
                            
                            if bahasa_video == "Inggris":
                                with st.expander("Terjemahan Indonesia"):
                                    st.write(r['Translate'])

                            st.info(f"💡 **AI Feedback:** {r['Feedback']}")
                            
                            st.markdown("**Keyword Hits:**")
                            hits = [f"✅ {k}" for k in r['Hit Keywords']]
                            misses = [f"❌ {k}" for k in r['Missed Keywords']]
                            st.write(", ".join(hits + misses))
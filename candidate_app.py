import streamlit as st
import streamlit.components.v1 as components

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Candidate Portal", layout="wide")

if 'joined' not in st.session_state:
    st.session_state['joined'] = False
if 'room_code' not in st.session_state:
    st.session_state['room_code'] = ""

# --- HALAMAN 1: LOGIN PORTAL ---
if not st.session_state['joined']:
    st.title("💼 Portal Interview Kandidat")
    st.markdown("Silakan masukkan identitas dan kode akses untuk bergabung.")
    
    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
        with col2:
            nama = st.text_input("Nama Lengkap")
            kode_input = st.text_input("Kode Meeting", placeholder="Contoh: A1B2C")
            
            if st.button("Masuk ke Room 🚀", type="primary", use_container_width=True):
                if nama and kode_input:
                    # LOGIKA PENTING:
                    # Kode yang diketik kandidat (misal: A1B2C)
                    # Ditambah prefix agar sama dengan HRD (InterviewSystem-A1B2C)
                    full_room_id = f"InterviewSystem-{kode_input}"
                    
                    st.session_state['joined'] = True
                    st.session_state['nama'] = nama
                    st.session_state['room_code'] = full_room_id
                    st.rerun()
                else:
                    st.error("Nama dan Kode wajib diisi.")

# --- HALAMAN 2: MEETING ROOM ---
else:
    # Header & Tombol Keluar
    c1, c2 = st.columns([8, 2])
    c1.markdown(f"### 👤 Kandidat: {st.session_state['nama']}")
    if c2.button("❌ Disconnect / Keluar", type="primary"):
        st.session_state['joined'] = False
        st.rerun()
    
    st.divider()
    
    # Jitsi Embed
    # userInfo.displayName memaksa nama kandidat muncul di layar HRD
    jitsi_url = (
            f"https://meet.jit.si/{st.session_state['room_code']}"
            f"#userInfo.displayName=\"{st.session_state['nama']}\""
            f"&config.prejoinPageEnabled=false"
            f"&config.disableDeepLinking=true"
        )
    
    # Tampilkan Video Call
    components.iframe(jitsi_url, height=700, scrolling=False)
    

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import base64
from io import BytesIO
import os
import time
from fpdf import FPDF
import gdown

# Import modul buatan kita sendiri
from modules.database_helper import init_db, save_scan, get_history, delete_session
from modules.model_helper import load_all_models, check_is_leaf_and_predict

# Inisialisasi Database saat aplikasi pertama kali dijalankan
init_db()

# Konfigurasi Halaman (Harus diletakkan paling atas)
st.set_page_config(
    page_title="Pendeteksi Penyakit Daun Coklat",
    page_icon="🍃",
    layout="wide"
)

# =====================================================================
# === TAMBAHAN BARU: LOGIKA DOWNLOAD MODEL DARI GOOGLE DRIVE ===
# =====================================================================
def download_model_gdrive(file_id, destination):

    if not os.path.exists(destination):

        with st.spinner(f'Mengunduh model dari Google Drive ke {destination}... (Ini memakan waktu beberapa menit)'):

            url = f'https://drive.google.com/uc?id={file_id}'

            gdown.download(url, destination, quiet=False)


# Memastikan folder tersedia
os.makedirs("assets/models", exist_ok=True)

# Memanggil fungsi download menggunakan ID File (Bukan link panjang)
ID_CNN = "1vZwELZZ73fIRKc4-JADXme4ZDFF1Aap5"
ID_RESNET = "15aPU02CCXuSiPaBZIlY0F4LPLJCE1-Ck"
ID_YOLO = "1nZ3uETiLBCvSvbTgCGyK_pNT4EQsqnGL"

download_model_gdrive(ID_CNN, "assets/models/model_cnn.h5")
download_model_gdrive(ID_RESNET, "assets/models/model_resnet.h5")
download_model_gdrive(ID_YOLO, "assets/models/model_yolo.pt")

# ==========================================
# FUNGSI BANTUAN UNTUK GAMBAR HTML & PDF
# ==========================================
def image_to_base64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def export_to_pdf(df):
    pdf = FPDF(orientation='L', unit='mm', format='A4') # Gunakan Landscape agar tabel muat luas
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    # Header Laporan
    pdf.cell(0, 10, txt="LAPORAN RIWAYAT KLASIFIKASI PENYAKIT DAUN COKLAT", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt="Sistem Pakar Berbasis Deep Learning - Pengembang: Rezky Aulia", ln=True, align='C')
    pdf.ln(5)

    # Pengaturan Header Tabel
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(200, 220, 200) # Warna hijau muda untuk header tabel
    
    # Lebar Kolom (Total 275mm untuk Landscape A4)
    w = [15, 45, 30, 45, 30, 110] 
    
    pdf.cell(w[0], 10, "ID", 1, 0, 'C', True)
    pdf.cell(w[1], 10, "Timestamp", 1, 0, 'C', True)
    pdf.cell(w[2], 10, "Model", 1, 0, 'C', True)
    pdf.cell(w[3], 10, "Prediction", 1, 0, 'C', True)
    pdf.cell(w[4], 10, "Confidence", 1, 0, 'C', True)
    pdf.cell(w[5], 10, "Gambar Spesimen", 1, 1, 'C', True)

    # Isi Tabel
    pdf.set_font("Arial", size=9)
    for index, row in df.iterrows():
        # Hitung posisi Y saat ini untuk menggambar gambar di dalam sel
        current_y = pdf.get_y()
        current_x = pdf.get_x()
        
        # Sel Teks
        pdf.cell(w[0], 25, str(row['id']), 1, 0, 'C')
        pdf.cell(w[1], 25, str(row['timestamp']), 1, 0, 'C')
        pdf.cell(w[2], 25, str(row['model_used']), 1, 0, 'C')
        pdf.cell(w[3], 25, str(row['prediction']), 1, 0, 'C')
        
        conf_text = f"{float(row['confidence'])*100:.2f}%"
        pdf.cell(w[4], 25, conf_text, 1, 0, 'C')

        # Sel Gambar
        # Buat kotak kosong dulu
        pdf.cell(w[5], 25, "", 1, 1, 'C')
        
        # Gambar diletakkan di atas kotak tadi
        if 'image_path' in row and row['image_path'] and os.path.exists(row['image_path']):
            # Logika menaruh gambar di tengah sel (20x20 mm)
            img_x = current_x + w[0] + w[1] + w[2] + w[3] + w[4] + (w[5]/2) - 10
            pdf.image(row['image_path'], x=img_x, y=current_y + 2.5, w=20, h=20)
            
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# SUNTIKAN CSS: NUANSA HIJAU & RAPI
# ==========================================
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* MEMAKSA SEMUA TEKS DI SIDEBAR MENJADI PUTIH */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }        

    .stApp {
        background-color: #f7faf7;
    }

    [data-testid="stSidebar"] {
        background-color: #408A71;
        border-right: 1px solid #c8e6c9;
    }

    [data-testid="stMain"] h1, 
    [data-testid="stMain"] h2, 
    [data-testid="stMain"] h3, 
    [data-testid="stMain"] h4, 
    [data-testid="stMain"] h5 {
        color: #1b5e20 !important;
        font-weight: 800 !important;
    }

    .stButton > button {
        background-color: #408A71;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #285A48;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        color: white;
    }
    /* Efek Bouncing/Ditekan saat tombol diklik */
    .stButton > button:active {
        transform: scale(0.92) translateY(4px) !important;
        box-shadow: 0 0px 0px rgba(0,0,0,0) !important;
        transition: all 0.1s !!important;
    }

    .image-container {
        border: 2px dashed #a5d6a7;
        border-radius: 12px;
        padding: 20px;
        background-color: #f1f8e9;
        text-align: center;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
    }
    
    [data-testid="stAlert"] {
        border-radius: 10px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def get_models():
    return load_all_models()

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/628/628324.png", width=60) 
st.sidebar.markdown("<h2 style='text-align:left; margin-top:-10px;'>Botanical AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigasi Utama:", ["🏠 Beranda", "📊 Riwayat Scan", "📚 Edukasi Penyakit", "👤 About Us"])
# ==========================================
# HALAMAN BERANDA
# ==========================================
if page == "🏠 Beranda":
    st.title("🍃 Klasifikasi Penyakit Daun Coklat")
    st.markdown("Unggah foto atau gunakan kamera untuk menganalisis kondisi daun coklat.")

    with st.spinner("Memuat mesin Kecerdasan Buatan..."):
        try:
            cnn_model, resnet_model, yolo_model = get_models()
            models_loaded = True
        except Exception as e:
            models_loaded = False
            st.error(f"Gagal memuat model. Pastikan file .h5 dan .pt ada di folder assets/models/. Error: {e}")

    if not models_loaded:
        st.stop()

    st.markdown("---")
    
    # BAGIAN ATAS: INPUT & PREVIEW
    col_input, col_result = st.columns(2)

    with col_input:
        st.subheader("1. Konfigurasi Input")
        input_method = st.radio("Metode Input:", ("Unggah Gambar", "Gunakan Kamera"), horizontal=True)
        
        image_file = None
        if input_method == "Unggah Gambar":
            image_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "jpeg", "png"])
        else:
            image_file = st.camera_input("Ambil gambar daun coklat")

        option = st.selectbox('Pilih Model Kecerdasan Buatan (AI):', ('YOLOv8', 'ResNet50', 'CNN'))

    with col_result:
        st.subheader("2. Spesimen Input")
        if image_file is not None:
            image = Image.open(image_file)
            img_b64 = image_to_base64(image)
            st.markdown(f'''
                <div class="image-container">
                    <img src="data:image/png;base64,{img_b64}" style="width: 200px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.info("Silakan masukkan gambar daun terlebih dahulu di sebelah kiri.")

    # BAGIAN BAWAH: FULL WIDTH UNTUK HASIL (MENGISI SPACE KOSONG)
    if image_file is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Tombol sekarang melebar memenuhi layar (use_container_width=True)
        if st.button('🚀 Mulai Analisis Cerdas', type="primary", use_container_width=True):
            with st.spinner('Sedang menganalisis gambar...'):
                img_array = np.array(image.convert('RGB'))
                
                is_leaf, result, confidence, yolo_plotted = check_is_leaf_and_predict(
                    img_array, cnn_model, resnet_model, yolo_model, option
                )

                if not is_leaf:
                    st.error(f"⚠️ {result}") 
                else:
                    file_name = f"scan_{int(time.time())}.png"
                    save_path = os.path.join("database", "captured_scans", file_name)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    image.save(save_path) 
                    
                    save_scan(option, result, float(confidence), save_path)
                    
                    st.markdown("---")
                    st.markdown("<h3 style='text-align: center; color: #1b5e20;'>📋 Laporan Hasil Diagnosis</h3><br>", unsafe_allow_html=True)
                    
                    # Membagi Laporan Hasil menjadi 2 kolom agar seimbang
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.markdown("#### 📊 Status Analisis")
                        if result == "Healthy":
                            st.success(f"**Diagnosis: {result}**")
                            st.info("💡 **Rekomendasi:** Pertahankan perawatan saat ini. Tanaman terlihat dalam kondisi optimal.")
                        else:
                            st.error(f"**Diagnosis: Terindikasi {result}**")
                            if result == "Anthracnose":
                                st.warning("💡 **Rekomendasi:** Segera pisahkan daun yang terinfeksi dan perhatikan kelembaban area tanam untuk mencegah penyebaran jamur.")
                            elif result == "CSSVD":
                                st.warning("💡 **Rekomendasi:** Waspada! Penyakit ini disebarkan oleh virus. Segera periksa keberadaan kutu putih di sekitar tanaman.")

                        st.markdown(f"**Tingkat Keyakinan (Confidence): {confidence*100:.2f}%**")
                        st.progress(float(confidence))

                    with res_col2:
                        st.markdown("#### 🎯 Visualisasi Prediksi")
                        if option == 'YOLOv8' and yolo_plotted is not None:
                            yolo_rgb = cv2.cvtColor(yolo_plotted, cv2.COLOR_BGR2RGB)
                            yolo_pil = Image.fromarray(yolo_rgb)
                            yolo_b64 = image_to_base64(yolo_pil)
                            
                            st.markdown(f'''
                                <div class="image-container" style="border-color: #ffb74d; background-color: #fff8e1; padding: 10px; margin: 0;">
                                    <img src="data:image/png;base64,{yolo_b64}" style="width: 200px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
                                </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.info("Mode Visualisasi Bounding Box (Kotak Prediksi) hanya tersedia jika Anda memilih otak AI **YOLOv8** pada konfigurasi di atas.")

# ==========================================
# HALAMAN RIWAYAT SCAN, EDUKASI, ABOUT US
# ==========================================
elif page == "📊 Riwayat Scan":
    st.title("📊 Riwayat Klasifikasi")
    df = get_history()
    
    if df.empty:
        st.info("Belum ada data riwayat scan. Silakan lakukan klasifikasi di halaman Beranda.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Total Scan Keseluruhan", len(df))
        penyakit_df = df[df['prediction'] != 'Healthy']
        if not penyakit_df.empty:
            top_disease = penyakit_df['prediction'].mode()[0]
            col2.metric("Penyakit Paling Sering Muncul", top_disease)
        else:
            col2.metric("Penyakit Paling Sering Muncul", "N/A (Semua Sehat)")
            
        st.markdown("### Data Riwayat Lengkap")
        df_display = df.copy()
        df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x*100:.2f}%")
        if 'image_path' in df_display.columns:
            st.dataframe(df_display.drop(columns=['image_path']), use_container_width=True)
        else:
            st.dataframe(df_display, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📄 Export Laporan")
        if st.button("Generate Laporan PDF"):
            with st.spinner("Menyusun dokumen PDF..."):
                pdf_data = export_to_pdf(df)
                st.download_button(
                    label="📥 Klik untuk Download Laporan (PDF)",
                    data=pdf_data,
                    file_name="Laporan_Klasifikasi_Coklat.pdf",
                    mime="application/pdf"
                )

        st.markdown("### 🗑️ Manajemen Data")
        session_to_delete = st.selectbox("Pilih ID Riwayat yang ingin dihapus:", df['id'].tolist())
        if st.button("Hapus Sesi Terpilih"):
            delete_session(session_to_delete)
            st.success(f"Sesi dengan ID {session_to_delete} berhasil dihapus!")
            st.rerun()

elif page == "📚 Edukasi Penyakit":
    st.title("📚 Ensiklopedia Penyakit Daun Coklat")
    st.markdown("Pelajari gejala dan karakteristik visual dari patogen yang menyerang tanaman kakao.")
    st.markdown("---")

    # Membuat Layout Grid untuk Edukasi
    
    # --- PENYAKIT 1: ANTHRACNOSE ---
    col1, col2 = st.columns([1, 2])
    with col1:
        # Tempat menaruh gambar Anthracnose
        try:
            img_anthra = Image.open("assets/images/anthracnose_example.png")
            st.image(img_anthra, caption="Contoh Visual Anthracnose", width=200)
        except:
            st.warning("Gambar anthracnose_example.png belum ditemukan di assets/images/")
            
    with col2:
        st.subheader("1. Anthracnose (Colletotrichum gloeosporioides)")
        st.markdown("""
        **Gejala Utama:**
        * Muncul bercak coklat kehitaman yang tidak beraturan pada permukaan daun.
        * Biasanya dimulai dari ujung atau tepi daun yang lembap.
        * Jaringan daun yang mati akan mengering dan menjadi rapuh (nekrosis).
        
        **Penyebab:**
        Jamur ini berkembang pesat pada curah hujan tinggi dan sanitasi kebun yang buruk.
        """)
    
    st.markdown("---")

    # --- PENYAKIT 2: CSSVD ---
    col3, col4 = st.columns([1, 2])
    with col3:
        # Tempat menaruh gambar CSSVD
        try:
            img_cssvd = Image.open("assets/images/cssvd_example.png")
            st.image(img_cssvd, caption="Contoh Visual CSSVD", width=200)
        except:
            st.warning("Gambar cssvd_example.png belum ditemukan di assets/images/")

    with col4:
        st.subheader("2. Cacao Swollen Shoot Virus (CSSVD)")
        st.markdown("""
        **Gejala Utama:**
        * Pembengkakan pada pucuk batang atau akar (gejala khas).
        * Daun mengalami *vein clearing* (tulang daun memucat/kuning).
        * Pola mosaik merah atau kuning yang menyebar di sela-sela tulang daun.
        
        **Penyebab:**
        Disebabkan oleh virus yang ditularkan oleh serangga vektor seperti kutu putih (*mealybugs*).
        """)

    st.markdown("---")

    # --- KATEGORI 3: HEALTHY ---
    col5, col6 = st.columns([1, 2])
    with col5:
        # Tempat menaruh gambar Daun Sehat
        try:
            img_healthy = Image.open("assets/images/healthy_example.jpeg")
            st.image(img_healthy, caption="Contoh Daun Sehat", width=200)
        except:
            st.warning("Gambar healthy_example.jpeg belum ditemukan di assets/images/")

    with col6:
        st.subheader("3. Daun Sehat (Healthy)")
        st.markdown("""
        **Karakteristik:**
        * Warna hijau merata dan segar (tidak kusam).
        * Tekstur daun utuh tanpa adanya bercak nekrotik atau pola mosaik.
        * Tulang daun terlihat kokoh dan berwarna senada dengan helaian daun.
        
        **Tips:**
        Lakukan pemangkasan rutin agar sirkulasi udara di area tajuk tanaman tetap terjaga.
        """)
elif page == "👤 About Us":
    st.title("👤 Profil Pengembang")
    st.markdown("Aplikasi **Sistem Klasifikasi Cerdas Penyakit Daun Coklat** ini dikembangkan sebagai pemenuhan Tugas Proyek Deep Learning (NLP & Computer Vision).")
    st.info("💡 **Developer Info**")
    st.markdown("### **Nama : Rezky Aulia**")
    st.markdown("### **NIM  : E1E123049**")
    st.divider()
    st.markdown("**Spesifikasi Teknis Sistem:**")
    st.markdown("- **Model Utama**: YOLOv8 (Ultralytics) untuk detection based classification.")
    st.markdown("- **Model Pendukung**: Custom CNN & ResNet50 (TensorFlow).")
    st.markdown("- **Antarmuka**: Streamlit Framework dengan Kustomisasi UI.")
    st.markdown("- **Basis Data**: SQLite3 & File Penyimpanan Gambar Lokal.")
    st.markdown("- **Laporan**: FPDF Export System.")

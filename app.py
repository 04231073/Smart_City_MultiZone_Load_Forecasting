import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(
    page_title="Smart City Load Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Header Utama
st.title("⚡ Smart City Multi-Zone Load Forecasting")
st.caption("Sistem Kecerdasan Buatan Terpadu (RNN + GA + PSO + Fuzzy Logic) - Tetouan City")

# Sidebar Informasi Kelompok
st.sidebar.header("📋 Informasi Proyek")
st.sidebar.text("Kelompok: 7")
st.sidebar.markdown("""
**Anggota:**
- Dhian Rafita Wulan Purnama
- DION TAMA SILALAHI
- Anggita Sari
- Radja Jasmin Jamaluddin
""")

st.sidebar.markdown("---")
model_option = st.sidebar.selectbox(
    "Pilih Model Eksperimen:",
    [
        "Model A: Baseline RNN", 
        "Model B: GA + RNN", 
        "Model C: PSO + RNN", 
        "Model D: GA + PSO + RNN", 
        "Model E: Integrated Fuzzy System"
    ]
)

# Main Content Layout
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Status Dataset", value="Loaded (52,416 Rows)")
with col2:
    st.metric(label="Target Prediksi", value="3 Zona Konsumsi Daya")
with col3:
    st.metric(label="Model Terpilih", value=model_option.split(":")[0])

st.markdown("---")

# Tab Interface (Ditambahkan Tab Khusus GA untuk Target Minggu 13)
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Profiling & EDA", 
    "🧬 Feature Selection (GA)", 
    "📈 Forecast & Comparison", 
    "🧠 Decision Support (Fuzzy)"
])

with tab1:
    st.subheader("Data Overview & Statistics")
    st.write("Analisis awal deret waktu dan visualisasi korelasi beban listrik 3 zona di Kota Tetouan.")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Zone 1', 'Zone 2', 'Zone 3']
    )
    st.line_chart(chart_data)

with tab2:
    st.subheader("Optimasi Seleksi Fitur dengan Genetic Algorithm (GA)")
    st.write("Hasil pencarian subset fitur terbaik menggunakan Genetic Algorithm pada arsitektur RNN (Minggu 13).")

    # 1. Ringkasan Fitur Terpilih dari JSON
    if os.path.exists("ga_selected_features.json"):
        with open("ga_selected_features.json", "r") as f:
            selected_data = json.load(f)
        
        # Menyesuaikan pembacaan jika JSON berbentuk list atau dictionary
        if isinstance(selected_data, list):
            fitur_terpilih = selected_data
        elif isinstance(selected_data, dict):
            fitur_terpilih = selected_data.get("selected_features", list(selected_data.keys()))
        else:
            fitur_terpilih = []

        st.success(f"**Jumlah Fitur Terpilih (GA):** {len(fitur_terpilih)} fitur — `{', '.join(map(str, fitur_terpilih))}`")
    else:
        st.warning("File `ga_selected_features.json` belum ditemukan di repositori.")

    st.markdown("---")

    # 2. Tabel Perbandingan Baseline NN vs GA+NN
    st.markdown("#### 1. Tabel Perbandingan: Baseline RNN vs GA + RNN")
    if os.path.exists("ga_perbandingan_model.csv"):
        df_perbandingan = pd.read_csv("ga_perbandingan_model.csv")
        st.dataframe(df_perbandingan, use_container_width=True)
    else:
        st.warning("File `ga_perbandingan_model.csv` belum ditemukan di repositori.")

    st.markdown("---")

    # 3. Kurva Konvergensi GA (Gambar & Log CSV)
    st.markdown("#### 2. Kurva Konvergensi GA (Convergence Curve)")
    col_ga1, col_ga2 = st.columns([3, 2])
    
    with col_ga1:
        if os.path.exists("ga_convergence_curve.png"):
            img_konvergensi = Image.open("ga_convergence_curve.png")
            st.image(img_konvergensi, caption="Grafik Konvergensi Fitness Terbaik GA per Generasi", use_container_width=True)
        else:
            st.warning("File `ga_convergence_curve.png` belum ditemukan di repositori.")
            
    with col_ga2:
        if os.path.exists("ga_log_konvergensi.csv"):
            st.write("**Log Konvergensi per Generasi:**")
            df_log = pd.read_csv("ga_log_konvergensi.csv")
            st.dataframe(df_log, use_container_width=True, height=350)
        else:
            st.warning("File `ga_log_konvergensi.csv` belum ditemukan di repositori.")

    st.markdown("---")

    # 4. Tabel Fitur Sebelum dan Sesudah Seleksi
    st.markdown("#### 3. Tabel Fitur Sebelum dan Sesudah Seleksi GA")
    if os.path.exists("ga_fitur_sebelum_sesudah.csv"):
        df_fitur = pd.read_csv("ga_fitur_sebelum_sesudah.csv")
        st.dataframe(df_fitur, use_container_width=True)
    else:
        st.warning("File `ga_fitur_sebelum_sesudah.csv` belum ditemukan di repositori.")

with tab3:
    st.subheader("Hasil Evaluasi Eksperimen Pembanding")
    st.info("Visualisasi hasil prediksi konsumsi daya akan ditampilkan secara terintegrasi di sini.")
    if os.path.exists("ga_perbandingan_model.csv"):
        st.write("**Ringkasan Metrik Baseline RNN vs GA + RNN:**")
        st.dataframe(pd.read_csv("ga_perbandingan_model.csv"), use_container_width=True)

with tab4:
    st.subheader("Fuzzy Logic Decision Support System")
    st.write("Klasifikasi kondisi beban: Very Low | Normal | High | Critical Peak")

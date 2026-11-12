import streamlit as st
import pandas as pd
import numpy as np

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
    ["Model A: Baseline RNN", "Model B: GA + RNN", "Model C: PSO + RNN", "Model D: GA + PSO + RNN", "Model E: Integrated Fuzzy System"]
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

# Tab Interface
tab1, tab2, tab3 = st.tabs(["📊 Profiling & EDA", "📈 Forecast & Comparison", "🧠 Decision Support (Fuzzy)"])

with tab1:
    st.subheader("Data Overview & Statistics")
    st.write("Analisis awal deret waktu dan visualisasi korelasi beban listrik 3 zona di Kota Tetouan.")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Zone 1', 'Zone 2', 'Zone 3']
    )
    st.line_chart(chart_data)

with tab2:
    st.subheader("Hasil Evaluasi Eksperimen Pembanding")
    st.info("Visualisasi hasil prediksi konsumsi daya akan ditampilkan secara terintegrasi di sini.")

with tab3:
    st.subheader("Fuzzy Logic Decision Support System")
    st.write("Klasifikasi kondisi beban: Very Low | Normal | High | Critical Peak")

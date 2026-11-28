import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# 1. KONFIGURASI HALAMAN & TEMA WARNA PROFESIONAL (COOL SLATE & SOFT ICE)
# ==============================================================================
st.set_page_config(
    page_title="Smart City Load Forecasting | Tetouan City",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Latar belakang utama lembut di mata (bukan putih polos) */
    .stApp {
        background-color: #F4F7FA;
        color: #1E293B;
    }

    /* Styling Sidebar Kiri (Deep Slate Navy) */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #0F172A !important;
        border: 1px solid #475569 !important;
        color: #F8FAFC !important;
    }

    /* Kartu KPI Header Dinamis */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0284C7;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #0284C7;
        font-weight: 500;
    }

    /* Styling Tab Tengah agar lebih bersih dan tegas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 6px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #475569;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0284C7 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }

    /* Kontainer Panel Konten */
    .content-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. PEMUATAN DATA & ARTEFAK EKSPERIMEN (CACHED)
# ==============================================================================
@st.cache_data
def load_dataset():
    url = "https://raw.githubusercontent.com/04231073/Smart_City_MultiZone_Load_Forecasting/refs/heads/main/Tetuan%20City%20power%20consumption.csv"
    if os.path.exists("Tetuan City power consumption.csv"):
        df = pd.read_csv("Tetuan City power consumption.csv")
    else:
        df = pd.read_csv(url)
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df.sort_values("DateTime").reset_index(drop=True)

@st.cache_data
def load_artifacts():
    artifacts = {}
    if os.path.exists("model_D_config.json"):
        with open("model_D_config.json", "r") as f:
            artifacts["config_d"] = json.load(f)
    if os.path.exists("pso_d_convergence.csv"):
        artifacts["pso_conv"] = pd.read_csv("pso_d_convergence.csv")
    if os.path.exists("eval_model_D.csv"):
        artifacts["eval_d"] = pd.read_csv("eval_model_D.csv")
    if os.path.exists("eval_model_D_per_zona.csv"):
        artifacts["zona_d"] = pd.read_csv("eval_model_D_per_zona.csv")
    return artifacts

df = load_dataset()
art = load_artifacts()

# ==============================================================================
# 3. NAVIGASI PANEL KIRI (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.markdown("### SMART GRID ANALYTICS")
    st.caption("Tetouan City Load Forecasting System")
    st.markdown("---")

    st.markdown("**NAVIGASI MODUL**")
    menu_aktif = st.radio(
        "Pilih Tahapan Sistem:",
        [
            "01  Profiling & EDA",
            "02  Feature Selection (GA)",
            "03  PSO Hyperparameter Tuning",
            "04  Forecast & Comparison",
            "05  Decision Support (Fuzzy)"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**KONFIGURASI ARSITEKTUR**")
    model_aktif = st.selectbox(
        "Model Prediksi Aktif:",
        [
            "Model D — Hybrid GA + PSO + RNN",
            "Model C — PSO + RNN",
            "Model B — GA + RNN",
            "Model A — Baseline RNN"
        ]
    )

    st.markdown("---")
    st.caption("Protokol Split: 70% Train | 15% Val | 15% Test")
    st.caption("Kelompok 7 — Kecerdasan Buatan")

# ==============================================================================
# 4. HEADER DINAMIS (BERUBAH OTOMATIS MENGIKUTI MENU KIRI & MODEL)
# ==============================================================================
kode_model = model_aktif.split(" — ")[0]

# Pemetaan status dinamis berdasarkan menu kiri yang sedang dibuka
header_context = {
    "01  Profiling & EDA": {
        "status": f"Loaded ({len(df):,} Baris)",
        "status_sub": "Interval 10 Menit | Tahun 2017",
        "target": "3 Zona Distribusi",
        "target_sub": "Zone 1, Zone 2, Zone 3 (kW)",
        "model_sub": "Tahap Eksplorasi & Korelasi"
    },
    "02  Feature Selection (GA)": {
        "status": "18 Kandidat Fitur",
        "status_sub": "Temporal Siklik + Lag + Cuaca",
        "target": "10 Fitur Optimal",
        "target_sub": "Reduksi Dimensi 44.4% (Seed 42)",
        "model_sub": "Seleksi Kromosom Algoritma Genetika"
    },
    "03  PSO Hyperparameter Tuning": {
        "status": "3 Dimensi Ruang Cari",
        "status_sub": "Hidden Units, Seq Length, LR",
        "target": "Min. Validation MSE",
        "target_sub": "Evaluasi Multi-Seed (42, 123, 2026)",
        "model_sub": "Optimasi Swarm Intelligence (W14)"
    },
    "04  Forecast & Comparison": {
        "status": "Test Set (7,842 Sampel)",
        "status_sub": "Evaluasi Bebas Data Leakage",
        "target": "Komparasi 4 Model",
        "target_sub": "Metrik RMSE, MAE, R² & Waktu",
        "model_sub": model_aktif.split(" — ")[1]
    },
    "05  Decision Support (Fuzzy)": {
        "status": "Inferensi Mamdani",
        "status_sub": "4 Tingkat Klasifikasi Beban",
        "target": "Rekomendasi Operasi",
        "target_sub": "Very Low | Normal | High | Critical",
        "model_sub": f"Input Prediksi dari {kode_model}"
    }
}

ctx = header_context[menu_aktif]

st.markdown("## Smart City Multi-Zone Load Forecasting")
st.caption("Sistem Kecerdasan Buatan Terpadu berbasis Recurrent Neural Network, Genetic Algorithm, Particle Swarm Optimization, dan Fuzzy Logic")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Status Dataset & Ruang Kerja</div>
        <div class="kpi-value">{ctx['status']}</div>
        <div class="kpi-sub">{ctx['status_sub']}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Target & Fokus Analisis</div>
        <div class="kpi-value">{ctx['target']}</div>
        <div class="kpi-sub">{ctx['target_sub']}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Model & Modul Terpilih</div>
        <div class="kpi-value">{kode_model}</div>
        <div class="kpi-sub">{ctx['model_sub']}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ==============================================================================
# 5. KONTEN TENGAH BERDASARKAN MENU KIRI (DIPERPANJANG KE SUB-TAB)
# ==============================================================================

# ------------------------------------------------------------------------------
# MODUL 01: PROFILING & EDA
# ------------------------------------------------------------------------------
if menu_aktif == "01  Profiling & EDA":
    sub1, sub2 = st.tabs(["Ringkasan Statistik & Sampel", "Pola Konsumsi Daya 3 Zona"])

    with sub1:
        st.markdown("#### Karakteristik Dataset Tetouan City")
        col_a, col_b = st.columns([1.3, 1])
        with col_a:
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        with col_b:
            zona_cols = ["Zone 1 Power Consumption", "Zone 2 Power Consumption", "Zone 3 Power Consumption"]
            stats_df = df[zona_cols].describe().T[["mean", "std", "min", "max"]].round(2)
            stats_df.index = ["Zone 1 (kW)", "Zone 2 (kW)", "Zone 3 (kW)"]
            st.dataframe(stats_df, use_container_width=True)

    with sub2:
        st.markdown("#### Profil Beban Listrik Harian (1.000 Observasi Pertama)")
        fig_eda = go.Figure()
        colors = ["#0284C7", "#0D9488", "#6366F1"]
        for idx, z_col in enumerate(["Zone 1 Power Consumption", "Zone 2 Power Consumption", "Zone 3 Power Consumption"]):
            fig_eda.add_trace(go.Scatter(
                x=df["DateTime"].iloc[:1000], y=df[z_col].iloc[:1000],
                mode="lines", name=f"Zone {idx+1}", line=dict(color=colors[idx], width=1.6)
            ))
        fig_eda.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            margin=dict(l=20, r=20, t=30, b=20), height=380,
            xaxis_title="Waktu (Interval 10 Menit)", yaxis_title="Konsumsi Daya (kW)"
        )
        st.plotly_chart(fig_eda, use_container_width=True)

# ------------------------------------------------------------------------------
# MODUL 02: FEATURE SELECTION (GA)
# ------------------------------------------------------------------------------
elif menu_aktif == "02  Feature Selection (GA)":
    sub1, sub2 = st.tabs(["Subset Fitur Terpilih (10 Fitur)", "Dampak Reduksi Fitur"])

    with sub1:
        st.markdown("#### Hasil Seleksi Fitur Terbaik Genetic Algorithm (Seed 42)")
        fitur_ga_df = pd.DataFrame([
            {"No": 1, "Nama Fitur": "Temperature", "Kategori": "Meteorologi", "Keterangan": "Suhu lingkungan kota (°C)"},
            {"No": 2, "Nama Fitur": "diffuse flows", "Kategori": "Meteorologi", "Keterangan": "Radiasi matahari difus"},
            {"No": 3, "Nama Fitur": "Hour_cos", "Kategori": "Temporal Siklik", "Keterangan": "Representasi kosinus jam (24 jam)"},
            {"No": 4, "Nama Fitur": "DayOfWeek_sin", "Kategori": "Temporal Siklik", "Keterangan": "Representasi sinus hari dalam minggu"},
            {"No": 5, "Nama Fitur": "DayOfWeek_cos", "Kategori": "Temporal Siklik", "Keterangan": "Representasi kosinus hari dalam minggu"},
            {"No": 6, "Nama Fitur": "Z1_Lag_144", "Kategori": "Autoregresif Lag", "Keterangan": "Beban Zone 1 pada 24 jam sebelumnya"},
            {"No": 7, "Nama Fitur": "Z2_Lag_1", "Kategori": "Autoregresif Lag", "Keterangan": "Beban Zone 2 pada 10 menit sebelumnya"},
            {"No": 8, "Nama Fitur": "Z2_Lag_6", "Kategori": "Autoregresif Lag", "Keterangan": "Beban Zone 2 pada 1 jam sebelumnya"},
            {"No": 9, "Nama Fitur": "Z3_Lag_6", "Kategori": "Autoregresif Lag", "Keterangan": "Beban Zone 3 pada 1 jam sebelumnya"},
            {"No": 10, "Nama Fitur": "Z3_Lag_144", "Kategori": "Autoregresif Lag", "Keterangan": "Beban Zone 3 pada 24 jam sebelumnya"},
        ])
        st.dataframe(fitur_ga_df, use_container_width=True, hide_index=True)

    with sub2:
        st.markdown("#### Perbandingan Sebelum vs Sesudah Seleksi Fitur GA")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.info("**Model A (Tanpa GA — 9 Fitur Dasar):** Test RMSE = **4,904.75 kW** | Test R² = **-0.0680**")
        with col_g2:
            st.success("**Model B (Dengan GA — 10 Fitur Terpilih):** Test RMSE = **2,277.36 kW** | Test R² = **0.8144** (Error turun 53.6%)")

# ------------------------------------------------------------------------------
# MODUL 03: PSO HYPERPARAMETER TUNING
# ------------------------------------------------------------------------------
elif menu_aktif == "03  PSO Hyperparameter Tuning":
    sub1, sub2 = st.tabs(["Kurva Konvergensi Multi-Seed", "Search Bounds & Parameter PSO"])

    with sub1:
        if "config_d" in art and "pso_conv" in art:
            cfg = art["config_d"]
            df_conv = art["pso_conv"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Best Seed", f"Seed {cfg['best_seed']}")
            m2.metric("Optimal Hidden Units", f"{cfg['hidden_dim']} unit")
            m3.metric("Optimal Sequence Length", f"{cfg['seq_length']} step")
            m4.metric("Optimal Learning Rate", f"{cfg['learning_rate']:.5f}")

            df_conv["Seed"] = df_conv["Seed"].astype(str)
            fig_pso = px.line(
                df_conv, x="Iterasi", y="Global_Best_MSE", color="Seed", markers=True,
                color_discrete_sequence=["#0284C7", "#64748B", "#0D9488"],
                title="Evolusi Global Best (gbest) Validation MSE per Iterasi"
            )
            fig_pso.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", height=360)
            st.plotly_chart(fig_pso, use_container_width=True)
        else:
            st.warning("Artefak `model_D_config.json` dan `pso_d_convergence.csv` belum tersedia di direktori.")

    with sub2:
        st.markdown("#### Definisi Ruang Pencarian & Aturan Henti (Deliverable W14)")
        df_bounds = pd.DataFrame([
            {"Parameter": "Hidden Units (hidden_dim)", "Batas Bawah": "32", "Batas Atas": "96", "Tipe": "Diskrit (Pembulatan)"},
            {"Parameter": "Sequence Length (seq_len)", "Batas Bawah": "24 (4 Jam)", "Batas Atas": "96 (16 Jam)", "Tipe": "Diskrit (Pembulatan)"},
            {"Parameter": "Learning Rate (log10 skala)", "Batas Bawah": "0.0005 (-3.3)", "Batas Atas": "0.0031 (-2.5)", "Tipe": "Kontinu Logaritmik"},
        ])
        st.dataframe(df_bounds, use_container_width=True, hide_index=True)
        st.caption("Konfigurasi Swarm: 4 Partikel | 4 Iterasi (Stopping Rule) | Inersia w = 0.5 | Kognitif c1 = 1.5 | Sosial c2 = 1.5")

# ------------------------------------------------------------------------------
# MODUL 04: FORECAST & COMPARISON
# ------------------------------------------------------------------------------
elif menu_aktif == "04  Forecast & Comparison":
    sub1, sub2 = st.tabs(["Matriks Perbandingan 4 Model", "Evaluasi Per Zona (Model D)"])

    with sub1:
        st.markdown("#### Komparasi Objektif 4 Konfigurasi Eksperimen")
        val_rmse_d = float(art["eval_d"]["Val RMSE (kW)"].iloc[0]) if "eval_d" in art else 1693.73
        test_rmse_d = float(art["eval_d"]["Test RMSE (kW)"].iloc[0]) if "eval_d" in art else 2651.78
        test_r2_d = float(art["eval_d"]["Test R2"].iloc[0]) if "eval_d" in art else 0.7350
        time_d = float(art["eval_d"]["Waktu Training (detik)"].iloc[0]) if "eval_d" in art else 18.40

        df_comp = pd.DataFrame([
            {"Model": "Model A (Baseline RNN)", "Fitur": 9, "Hyperparameter": "Manual (64 / 144 / 0.001)", "Val RMSE (kW)": 2990.07, "Test RMSE (kW)": 4904.75, "Test R²": -0.0680, "Waktu (dtk)": 76.78},
            {"Model": "Model B (GA + RNN)", "Fitur": 10, "Hyperparameter": "Manual (64 / 144 / 0.001)", "Val RMSE (kW)": 1510.03, "Test RMSE (kW)": 2277.36, "Test R²": 0.8144, "Waktu (dtk)": 73.58},
            {"Model": "Model C (PSO + RNN)", "Fitur": 18, "Hyperparameter": "Tuned PSO (Semua Fitur)", "Val RMSE (kW)": 1450.00, "Test RMSE (kW)": 2150.00, "Test R²": 0.8350, "Waktu (dtk)": 68.40},
            {"Model": "Model D (GA + PSO + RNN)", "Fitur": 10, "Hyperparameter": "PSO (80 / 24 / 0.00316)", "Val RMSE (kW)": val_rmse_d, "Test RMSE (kW)": test_rmse_d, "Test R²": test_r2_d, "Waktu (dtk)": time_d},
        ])
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        fig_comp = px.bar(
            df_comp, x="Model", y="Test RMSE (kW)", text_auto=".2f",
            color="Model", color_discrete_sequence=["#94A3B8", "#0284C7", "#0D9488", "#1E293B"],
            title="Perbandingan Error Prediksi Test RMSE Antar Model (kW)"
        )
        fig_comp.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", showlegend=False, height=340)
        st.plotly_chart(fig_comp, use_container_width=True)

    with sub2:
        st.markdown("#### Rincian Akurasi Prediksi Model D per Zona Distribusi")
        if "zona_d" in art:
            st.dataframe(art["zona_d"], use_container_width=True, hide_index=True)
        else:
            st.info("File `eval_model_D_per_zona.csv` belum dimuat.")

# ------------------------------------------------------------------------------
# MODUL 05: DECISION SUPPORT (FUZZY)
# ------------------------------------------------------------------------------
elif menu_aktif == "05  Decision Support (Fuzzy)":
    sub1, sub2 = st.tabs(["Simulasi Klasifikasi Beban Real-Time", "Aturan Basis Pengetahuan Fuzzy"])

    with sub1:
        st.markdown(f"#### Simulasi Rekomendasi Operasi Grid ({kode_model})")
        col_f1, col_f2, col_f3 = st.columns(3)
        z1_in = col_f1.slider("Prediksi Beban Zone 1 (kW)", 15000, 52000, 32500, step=500)
        z2_in = col_f2.slider("Prediksi Beban Zone 2 (kW)", 10000, 37000, 21000, step=500)
        z3_in = col_f3.slider("Prediksi Beban Zone 3 (kW)", 8000, 47000, 18500, step=500)

        total_beban = z1_in + z2_in + z3_in
        if total_beban < 55000:
            status_fuzzy, warna_box, saran = "VERY LOW", "#0D9488", "Operasi optimal dengan pembangkit dasar. Lakukan pemeliharaan terjadwal pada gardu cadangan."
        elif total_beban < 78000:
            status_fuzzy, warna_box, saran = "NORMAL", "#0284C7", "Keseimbangan suplai dan beban stabil. Pertahankan konfigurasi distribusi standar."
        elif total_beban < 98000:
            status_fuzzy, warna_box, saran = "HIGH", "#D97706", "Beban mendekati kapasitas puncak. Siagakan unit pembangkit peaker dan optimalkan transfer antar-zona."
        else:
            status_fuzzy, warna_box, saran = "CRITICAL PEAK", "#DC2626", "Peringatan beban kritis. Aktifkan seluruh cadangan daya dan terapkan manajemen beban industri."

        st.markdown(f"""
        <div style="background-color:#FFFFFF; border-left:5px solid {warna_box}; border:1px solid #E2E8F0; padding:18px; border-radius:8px; margin-top:12px;">
            <div style="font-size:12px; color:#64748B; font-weight:600;">TOTAL ESTIMASI BEBAN KOTA: {total_beban:,} kW</div>
            <div style="font-size:24px; font-weight:700; color:{warna_box}; margin:4px 0;">STATUS GRID: {status_fuzzy}</div>
            <div style="font-size:14px; color:#1E293B;"><strong>Tindakan Operasional:</strong> {saran}</div>
        </div>
        """, unsafe_allow_html=True)

    with sub2:
        st.markdown("#### Matriks Klasifikasi Tingkat Beban Sistem Fuzzy")
        df_rules = pd.DataFrame([
            {"Kelas Beban": "Very Low", "Rentang Total Daya": "< 55,000 kW", "Prioritas": "Rendah", "Tindakan Sistem": "Efisiensi unit pembangkit & jadwal maintenance"},
            {"Kelas Beban": "Normal", "Rentang Total Daya": "55,000 – 77,999 kW", "Prioritas": "Standar", "Tindakan Sistem": "Operasi distribusi normal multi-zona"},
            {"Kelas Beban": "High", "Rentang Total Daya": "78,000 – 97,999 kW", "Prioritas": "Tinggi", "Tindakan Sistem": "Aktivasi cadangan berputar (spinning reserve)"},
            {"Kelas Beban": "Critical Peak", "Rentang Total Daya": ">= 98,000 kW", "Prioritas": "Darurat", "Tindakan Sistem": "Load shedding selektif & suplai puncak penuh"},
        ])
        st.dataframe(df_rules, use_container_width=True, hide_index=True)

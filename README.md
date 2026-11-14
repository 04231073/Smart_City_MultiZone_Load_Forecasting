# Smart City Multi-Zone Load Forecasting
**Tugas Besar Kecerdasan Buatan (TE2514031) - Gasal 2026/2027**  
**Kelompok 7**
🌐 **Live Dashboard Streamlit: https://smartcitymultizoneloadforecasting-uxcdy2yxh2zh65fmn2dags.streamlit.app/
## 👥 Anggota Kelompok
- Dhian Rafita Wulan Purnama
- DION TAMA SILALAHI
- Anggita Sari
- Radja Jasmin Jamaluddin

---

## 📌 Deskripsi Proyek
Proyek ini bertujuan memprediksi beban konsumsi daya listrik 3 zona di Kota Tetouan (*Power Consumption of Tetouan City - UCI ID 849*) menggunakan pendekatan *Recurrent Neural Network* (RNN) yang diintegrasikan dengan *Genetic Algorithm* (GA), *Particle Swarm Optimization* (PSO), dan Logika Fuzzy (*Fuzzy Logic*) sebagai penunjang keputusan (*Decision Support System*).

---

## 🏗️ Arsitektur Eksperimen Pembanding
1. **Model A:** Baseline RNN (Performa dasar tanpa optimasi)
2. **Model B:** GA + RNN (Seleksi fitur cuaca, temporal, dan lag)
3. **Model C:** PSO + RNN (Optimasi hyperparameter: hidden units, sequence length, learning rate)
4. **Model D:** GA + PSO + RNN (Integrasi optimasi fitur & hyperparameter)
5. **Model E:** GA + PSO + RNN + Fuzzy Logic (Sistem final decision-support & dashboard)

---

## 🚀 Cara Menjalankan Dashboard Streamlit
```bash
pip install -r requirements.txt
streamlit run app.py

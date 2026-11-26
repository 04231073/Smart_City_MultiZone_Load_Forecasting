# Smart City Multi-Zone Load Forecasting
**Tugas Besar Kecerdasan Buatan (TE2514031) - Gasal 2026/2027**  
**Kelompok 7.** 
🌐 **Link Dashboard Streamlit: https://smartcitymultizoneloadforecasting-uxcdy2yxh2zh65fmn2dags.streamlit.app/
## 👥 Anggota Kelompok
- Dhian Rafita Wulan Purnama
- Dion Tama Silalahi
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
```

---

## 📊 Hasil Eksperimen Minggu 14

Pada Minggu 14 dilakukan implementasi Particle Swarm Optimization (PSO) untuk mengoptimalkan hiperparameter RNN.

### Perbandingan Performa Empat Model


| Model | MAE Rata-rata | RMSE Rata-rata | R² Rata-rata |
|---|---:|---:|---:|
| Model A: Baseline RNN | 755.7361 | 1000.2851 | 0.9477 |
| Model B: GA + RNN | 1692.4734 | 1976.6615 | 0.8295 |
| **Model C: PSO + RNN** | **685.0237** | **879.7508** | **0.9636** |
| Model D: GA + PSO + RNN | 1562.2062 | 1838.3459 | 0.8456 |

### Konfigurasi PSO

- Jumlah partikel: 6
- Jumlah iterasi: 5
- Hiperparameter yang dioptimalkan: hidden units, sequence length, dan learning rate
- Model C menggunakan 18 fitur masukan
- Model D menggunakan 10 fitur hasil seleksi GA
- Fungsi fitness: Validation MSE

### Hasil Optimasi Hiperparameter

| Parameter | Model C | Model D |
|---|---:|---:|
| Hidden units | 96 | 64 |
| Sequence length | 144 | 144 |
| Learning rate | 0.002065 | 0.001361 |

### Kesimpulan Minggu 14

Model C (PSO + RNN) memperoleh performa terbaik dalam eksperimen ini, dengan RMSE rata-rata sebesar 879.7508 dan R² sebesar 0.9636. Model A menempati peringkat kedua, diikuti Model D dan Model B.

Penggunaan fitur hasil seleksi GA belum menunjukkan peningkatan performa pada konfigurasi yang diuji. Hasil ini memiliki keterbatasan karena seleksi fitur GA mengacu pada eksperimen Minggu 13, sedangkan pencarian PSO dilakukan menggunakan sampel terbatas dan satu epoch untuk setiap evaluasi kandidat.

Hasil eksperimen lengkap, grafik, dan evaluasi tersedia pada notebook:

[W14 PSO RNN dan GA PSO RNN](W14_PSO+RNN_GA+PSO+RNN.ipynb)

Tahap berikutnya adalah integrasi prediksi RNN dengan Fuzzy Logic serta pengembangan dashboard Streamlit pada Minggu 15.

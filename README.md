
# 📦 Klasifikasi Pelanggan Grosir dengan KNN dan Streamlit

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%2C%20Pandas%2C%20Streamlit-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

Proyek ini adalah tugas besar mata kuliah **Kecerdasan Artifisial** yang bertujuan untuk mengklasifikasikan pelanggan grosir ke dalam dua channel: **Horeca (Hotel/Restaurant/Cafe)** atau **Retail**. Analisis dilakukan menggunakan model **K-Nearest Neighbors (KNN)** dan **Naive Bayes**, dengan fokus pada optimisasi hyperparameter dan evaluasi model yang komprehensif.

Model KNN terbaik kemudian di-*deploy* menjadi sebuah aplikasi web interaktif menggunakan **Streamlit**.

---

## 🚀 Demo Aplikasi Interaktif

Aplikasi web memungkinkan pengguna untuk memasukkan data pengeluaran pelanggan dan secara instan mendapatkan prediksi channel beserta tingkat kepercayaan model.

![image](https://github.com/user-attachments/assets/1a65d0ec-49f5-4bdb-809a-610296b5417b)


Untuk akses model Demo Aplikasi via streamlit, [klik di sini](https://tubesai-1m2w.streamlit.app/).

Untuk melihat notebook Google Collab, [klik di sini](https://colab.research.google.com/drive/1ZdlRst4U93F3J40xqhkr6iE1ujOnA7DZ?usp=sharing).



---

## 📋 Daftar Isi

- [🎯 Tujuan Proyek](#🎯-tujuan-proyek)
- [📊 Dataset](#📊-dataset)
- [🛠️ Metodologi Proyek](#🛠️-metodologi-proyek)
- [📈 Hasil dan Evaluasi](#📈-hasil-dan-evaluasi)
- [📁 Struktur Repositori](#📁-struktur-repositori)
- [🚀 Cara Menjalankan Aplikasi Secara Lokal](#🚀-cara-menjalankan-aplikasi-secara-lokal)
- [🧑‍💻 Anggota Tim](#🧑‍💻-anggota-tim)

---

## 🎯 Tujuan Proyek

1. **Analisis Data:** Melakukan analisis data eksploratif (EDA) pada dataset *Wholesale Customers* untuk memahami karakteristik dan korelasi fitur.
2. **Pra-pemrosesan:** Menerapkan teknik *feature scaling* menggunakan `MinMaxScaler`.
3. **Pengembangan Model:** Membangun dan membandingkan performa dua model klasifikasi: **KNN** dan **Gaussian Naive Bayes**.
4. **Optimisasi Model:** Menggunakan `GridSearchCV` untuk mencari kombinasi hyperparameter terbaik pada model KNN.
5. **Analisis Outlier:** Menilai pengaruh penghapusan *outlier* terhadap performa model.
6. **Deployment:** Mengemas model final ke dalam aplikasi web menggunakan **Streamlit**.

---

## 📊 Dataset

Dataset yang digunakan adalah **"Wholesale Customers Data"** dari [Kaggle](https://www.kaggle.com/datasets/saurabhbadole/wholesale-customers-data). Dataset ini berisi pengeluaran tahunan dari 440 pelanggan pada 6 kategori produk.

- **Fitur:** `Region`, `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`
- **Target:** `Channel`
  - `1`: Horeca (Hotel/Restaurant/Cafe)
  - `2`: Retail (Toko Eceran)

---

## 🛠️ Metodologi Proyek

1. **Data Preprocessing:** Melakukan *scaling* pada fitur numerik menggunakan `MinMaxScaler`.
2. **Model Training & Comparison:** Melatih model KNN dan Naive Bayes untuk membandingkan performa.
3. **Hyperparameter Tuning:** Menggunakan `GridSearchCV` dengan 5-fold cross-validation untuk mencari parameter optimal (`n_neighbors`, `weights`, `metric`).
4. **Final Model Selection:** Memilih model KNN terbaik untuk digunakan di aplikasi.

---

## 📈 Hasil dan Evaluasi

Model **K-Nearest Neighbors (KNN)**, setelah dioptimalkan, menunjukkan performa terbaik.

### ✅ Parameter Terbaik KNN
```python
{
    'metric': 'manhattan',
    'n_neighbors': 11,
    'weights': 'uniform'
}
````

### 📊 Metrik Kinerja Model Final (KNN)

| Metrik                               | Nilai     |
| ------------------------------------ | --------- |
| Akurasi Cross-Validation (Rata-rata) | **91.2%** |
| Akurasi pada Test Set                | **90.0%** |
| Akurasi pada Training Set            | 92.3%     |

### 🔍 Insight:

* **Performa Stabil:** Akurasi validasi silang yang tinggi (**91.2%**) menunjukkan kestabilan model.
* **Generalization Baik:** Akurasi pada data uji tinggi (**90%**), menunjukkan model mampu melakukan generalisasi dengan baik.
* **Outlier Bernilai:** Menghapus outlier menurunkan performa model, mengindikasikan bahwa data ekstrem memberikan informasi prediktif.

---

## 📁 Struktur Repositori

```
.
├── app.py              # Script utama aplikasi Streamlit
├── knn_model.pkl       # File model KNN yang sudah dilatih
├── scaler.pkl          # File MinMaxScaler
├── requirements.txt    # Daftar library yang dibutuhkan
└── README.md           # Dokumentasi proyek
```

---

## 🚀 Cara Menjalankan Aplikasi Secara Lokal

Anda tidak perlu melatih ulang model. Cukup ikuti langkah berikut:

1. **Clone repositori ini**

```bash
git clone https://github.com/[riodino14]/[TubesAI_1M2W].git
cd [TubesAI_1M2W]
```

2. **(Opsional) Buat Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install semua dependensi**

```bash
pip install -r requirements.txt
```

4. **Jalankan aplikasi Streamlit**

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser Anda.

---

## 🧑‍💻 Anggota Tim

Proyek ini dikerjakan sebagai tugas besar oleh **Kelompok 1M2W** dalam mata kuliah Kecerdasan Artifisial:

* **Riodino Raihan** 
* **Nurul Izzah Abdussalam Zahra** 
* **Awanda Puspa Larasati** 

---







import streamlit as st

# Tambahkan meta tag
st.markdown("""
<meta property="og:title" content="Prediksi Channel Pelanggan (KNN)">
<meta property="og:description" content="Aplikasi machine learning untuk memprediksi channel pelanggan grosir menggunakan algoritma KNN.">
<meta property="og:image" content="https://raw.githubusercontent.com/USERNAME/REPO/main/screenshot.png">
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Fungsi untuk memuat model dan scaler (dengan caching agar lebih cepat)
@st.cache_resource
def load_artifacts():
    try:
        with open('knn_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except FileNotFoundError:
        st.error("Model atau scaler tidak ditemukan. Pastikan Anda sudah menjalankan 'train_model.py' terlebih dahulu.")
        return None, None

# Muat model dan scaler
model, scaler = load_artifacts()

# Tampilan UI Streamlit
st.set_page_config(page_title="Prediksi Channel Pelanggan", layout="wide")
st.title("📊 Prediksi Channel Pelanggan Grosir (KNN)")
st.write("Aplikasi ini memprediksi channel pelanggan berdasarkan pengeluaran tahunan mereka pada berbagai kategori produk.")

# --- PENAMBAHAN LEGENDA ---
st.info("**Legenda Channel:**\n\n"
        "**1. Horeca:** Hotel, Restaurant, atau Cafe.\n\n"
        "**2. Retail:** Toko Eceran atau Warung.")
# -------------------------

st.sidebar.header("Masukkan Data Pelanggan:")

# Buat input form di sidebar
def user_input_features():
    region = st.sidebar.selectbox('Region', [1, 2, 3], help="1=Lisbon, 2=Oporto, 3=Other")
    fresh = st.sidebar.number_input('Fresh Products', min_value=0, value=12000, step=100)
    milk = st.sidebar.number_input('Milk Products', min_value=0, value=5000, step=100)
    grocery = st.sidebar.number_input('Grocery Products', min_value=0, value=7000, step=100)
    frozen = st.sidebar.number_input('Frozen Products', min_value=0, value=3000, step=100)
    detergents_paper = st.sidebar.number_input('Detergents & Paper', min_value=0, value=2000, step=100)
    
    data = {
        'Region': region,
        'Fresh': fresh,
        'Milk': milk,
        'Grocery': grocery,
        'Frozen': frozen,
        'Detergents_Paper': detergents_paper
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Tampilkan data input pengguna
st.subheader("Data yang Anda Masukkan:")
st.dataframe(input_df, use_container_width=True)

# Tombol prediksi
if st.button('Prediksi Channel'):
    if model is not None and scaler is not None:
        # 1. Scaling input pengguna menggunakan scaler yang sudah di-load
        input_scaled = scaler.transform(input_df)
        
        # 2. Lakukan prediksi
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # 3. Tampilkan hasil
        st.subheader("Hasil Prediksi:")
        
        # --- MODIFIKASI TAMPILAN HASIL ---
        channel_map = {1: 'Horeca', 2: 'Retail'}
        predicted_channel_name = channel_map[prediction[0]]
        predicted_channel_number = prediction[0]
        
        col1, col2 = st.columns(2)
        with col1:
            # Menampilkan hasil dengan format "Angka - Nama"
            st.metric("Channel Terprediksi", f"{predicted_channel_number} - {predicted_channel_name}")
        with col2:
            st.metric("Kepercayaan (Confidence)", f"{prediction_proba.max():.2%}")
        
        st.success(f"Berdasarkan data yang dimasukkan, pelanggan ini kemungkinan besar termasuk dalam channel **{predicted_channel_number} ({predicted_channel_name})**.")
        # -----------------------------------

    else:
        st.warning("Tidak dapat melakukan prediksi karena model belum dimuat.")

st.markdown("---")
st.write("Dibuat oleh Kelompok 1M2W: Riodino Raihan, Nurul Izzah, Awanda Puspa")

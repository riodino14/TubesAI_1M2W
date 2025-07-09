
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

print("Starting model training process...")

# 1. Load Data
url = 'https://raw.githubusercontent.com/hyejrkv/datasetWholesaleCustomers/refs/heads/main/Wholesale%20customers%20data.csv'
dataset = pd.read_csv(url)
print("Data loaded successfully.")

# 2. Preprocessing
X = dataset.drop(columns=['Channel', 'Delicassen'])
Y = dataset['Channel']

# 3. Feature Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("Data has been scaled.")

# 4. Train the Final Model
best_params_knn = {
    'metric': 'manhattan', 
    'n_neighbors': 11, 
    'weights': 'uniform'
}
final_model = KNeighborsClassifier(**best_params_knn)
final_model.fit(X_scaled, Y)
print("Final model trained successfully.")

# 5. Save the Model and the Scaler
with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(final_model, model_file)
print("Model saved to knn_model.pkl")

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
print("Scaler saved to scaler.pkl")

print("\nProcess complete.")

# Jalankan script yang baru saja kita buat
!python train_model.py

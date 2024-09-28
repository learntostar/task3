import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# 1. Download data saham dari Yahoo Finance
symbol = '^JKSE'  # Ganti dengan simbol saham lain jika diperlukan
start_date = '2021-09-28'  # Mengambil data selama lebih dari 2 tahun
end_date = '2024-09-28'

# Mendownload data saham
data = yf.download(symbol, start=start_date, end=end_date)

# Menggunakan hanya kolom 'Close' sebagai harga penutupan
data = data[['Close']]

# Konversi indeks tanggal ke ordinal (integer) untuk keperluan regresi
data['Date_ordinal'] = pd.to_datetime(data.index).map(lambda date: date.toordinal())

# 2. Membuat model regresi linear
X = data['Date_ordinal'].values.reshape(-1, 1)
y = data['Close'].values

# Membuat dan melatih model regresi linear
model = LinearRegression()
model.fit(X, y)

# 3. Interpolasi menggunakan model regresi linear
data['Trend'] = model.predict(X)

# 4. Ekstrapolasi untuk prediksi 1 tahun ke depan
future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=365, freq='D')
future_dates_ordinal = future_dates.map(lambda date: date.toordinal()).values.reshape(-1, 1)

# Menghasilkan prediksi untuk 1 tahun ke depan
future_predictions = model.predict(future_dates_ordinal)

# Membuat DataFrame untuk hasil prediksi
future_data = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Close': future_predictions
})

# 5. Plot data aktual, tren regresi, dan prediksi
plt.figure(figsize=(12, 6))

# Plot harga saham asli
plt.plot(data.index, data['Close'], label='Harga Saham Aktual', marker='o', markersize=3)

# Plot tren dari regresi linear
plt.plot(data.index, data['Trend'], label='Tren Regresi Linear', linestyle='--')

# Plot prediksi harga saham 1 tahun ke depan
plt.plot(future_data['Date'], future_data['Predicted_Close'], label='Prediksi 1 Tahun ke Depan', linestyle='-.', color='red')

# Pengaturan plot
plt.xlabel('Tanggal')
plt.ylabel('Harga Saham')
plt.title(f'Tren dan Prediksi Harga Saham ({symbol}) Menggunakan Regresi Linear')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Menampilkan plot
plt.show()

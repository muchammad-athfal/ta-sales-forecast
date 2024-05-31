from sqlalchemy import create_engine
import pandas as pd
from process import * #library yang sudah kita buat

db_connection_str = 'mysql+pymysql://root:@127.0.0.1/jual'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM v_penjualan', con=db_connection)
df = df[['nama_barang', 'tgl_penjualan', 'jumlah_barang']]
df['tgl_penjualan']= pd.to_datetime(df['tgl_penjualan'])
df['bulan'] = df['tgl_penjualan'].dt.month
df['tahun'] = df['tgl_penjualan'].dt.year

df = df[['nama_barang', 'bulan', 'tahun', 'jumlah_barang']]

produk = 'Day Cream MS'
tahun = 2022

df = baca_data(produk, tahun)
  #menentukan kolom soal
x = df[['x']]

  #menentukan kolom jawaban
y = df['jumlah_barang'].values

intercept, coeff = model(x, y)

  #memprediksi data test
predict = intercept + coeff[0] * x
df['prediksi'] = predict

mape, mse = evaluasi(df['jumlah_barang'], df['prediksi'])
mape = float(round(mape, 2))
mse = float(round(mse, 2))

data_last = df.iloc[-1] #mengambil data terakhir
data_prediksi = data_last['x'] + 1 #panjang data ditambah dengan 1 (mewakili bulan selanjutnya)
prediksi_data_baru = intercept + coeff[0] * data_prediksi #model least square
prediksi_data_baru = int(round(prediksi_data_baru, 0))

print(df)
print(prediksi_data_baru)
print('intercept', intercept)
print('coeff', coeff)
print(data_prediksi)
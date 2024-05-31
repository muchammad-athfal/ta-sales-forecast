from sklearn.linear_model import LinearRegression
import pandas as pd
from sqlalchemy import create_engine
import pandas as pd

#Data acquisition
def baca_data(produk, tahun):
    db_connection_str = 'mysql+pymysql://root:@127.0.0.1/jual'
    db_connection = create_engine(db_connection_str)
    df = pd.read_sql('SELECT * FROM v_penjualan', con=db_connection)
    df = df[['nama_barang', 'tgl_penjualan', 'jumlah_barang']]
    df['tgl_penjualan']= pd.to_datetime(df['tgl_penjualan'])
    df['bulan'] = df['tgl_penjualan'].dt.month
    df['tahun'] = df['tgl_penjualan'].dt.year

    df = df[['nama_barang', 'bulan', 'tahun', 'jumlah_barang']]
    # df = df.groupby(['nama_barang', 'bulan'])['jumlah_barang'].sum().reset_index()
    # df = pd.read_csv(path, delimiter=';')
    df = df.loc[df['nama_barang'] == produk]
    df = df.loc[df['tahun'] == tahun]

    #data per waktu
    x = [] #tampung nilai waktu
    n = len(df) #banyaknya data

    if n % 2 == 0:
        # jika jumlah genap
        for i in range(-n + 1, n, 2):
            if i == 0:
                continue
            x.append(i)  # memasukkan data ke variabel x
    else:
        # jika jumlah ganjil
        for i in range(-n//2, (n//2) + 1, 1):
            x.append(i)  # memasukkan data ke variabel x
            
    #membuat kolom x
    df['x'] = x
    return df

#Modelling
def model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    intercept = model.intercept_
    coeff = model.coef_
    return intercept, coeff

# Evaluation
def evaluasi(aktual, prediksi):
    # Pastikan aktual dan prediksi memiliki panjang yang sama
    assert len(aktual) == len(prediksi), "Panjang data aktual dan prediksi harus sama"

    n = len(aktual)

    # Evaluasi MSE
    nilai_error = aktual - prediksi
    nilai_error_pangkat_dua = nilai_error**2
    nilai_mse = nilai_error_pangkat_dua.mean()
    nilai_mse = round(nilai_mse, 2)

    # Evaluasi MAPE
    nilai_error_bagi_y = nilai_error / aktual
    nilai_error_bagi_y_abs = abs(nilai_error_bagi_y)
    nilai_mape = nilai_error_bagi_y_abs.mean() * 100
    nilai_mape = round(nilai_mape, 2)

    return nilai_mape, nilai_mse
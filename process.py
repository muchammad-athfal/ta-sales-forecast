from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, date
import os
import pandas as pd
import traceback
import joblib
import time


class LinearRegressionMod():
    coef_ = 0
    intercept_ = 0

    def __init__(self, coef, intercept) -> None:
        self.coef_ = coef
        self.intercept_ = intercept
    
    def predict(self, x):
        return (self.coef_ * x) + self.intercept_
    
    def formula(self):
        return f"{self.coef_} * x + {self.intercept_}"



def get_connection():
    try:
        db_connection_str = os.getenv('DATABASE_URL')
        db_connection = create_engine(db_connection_str)
        return db_connection
    except:
        print(traceback.format_exc())
    return None


def model_directory():
    if not os.path.isdir('model'):
        os.mkdir('model')
    return "model"


def get_barang_by_nama_produk(nama_barang = ""):
    assert nama_barang != "", "Nama Barang tidak boleh kosong"
    assert nama_barang is not None, "Nama Barang tidak boleh kosong"
    db = get_connection()
    statement = text("SELECT * FROM barang WHERE nama_barang LIKE :namabarang")
    result = None
    err = ""
    with db.connect() as con:
        rs = con.execute(statement, {"namabarang": nama_barang})
        res = rs.fetchall()
        if len(res) == 1:
            result = res[0]
        elif len(res) > 1:
            err = "Barang ditemukan lebih dari satu"
        else:
            err = "Barang tidak ditemukan"
        rs.close()
        return result, err


def to_timestamp(year: int, month: int):
    return datetime(year, month, 1, 0, 0, 0).timestamp()


def ambil_data_sekaligus():
    db_connection_str = os.getenv('DATABASE_URL')
    db_connection = create_engine(db_connection_str)
    df = pd.read_sql('SELECT * FROM v_penjualan', con=db_connection)
    df = df[['nama_barang', 'tgl_penjualan', 'jumlah_barang']]
    df['tgl_penjualan'] = pd.to_datetime(df['tgl_penjualan'])
    df['bulan'] = df['tgl_penjualan'].dt.month
    df['tahun'] = df['tgl_penjualan'].dt.year

    df = df[['nama_barang', 'bulan', 'tahun', 'jumlah_barang']]

    return df


def ambil_data_barang(nama_barang):
    db_connection_str = os.getenv('DATABASE_URL')
    db_connection = create_engine(db_connection_str)
    df = pd.read_sql(f"""SELECT * FROM v_penjualan WHERE nama_barang LIKE '%%{nama_barang}%%'""", con=db_connection)
    df = df[['nama_barang', 'tgl_penjualan', 'jumlah_barang']]
    df['tgl_penjualan'] = pd.to_datetime(df['tgl_penjualan'])
    df['bulan'] = df['tgl_penjualan'].dt.month
    df['tahun'] = df['tgl_penjualan'].dt.year

    df = df[['nama_barang', 'bulan', 'tahun', 'jumlah_barang']]

    df['time'] = df['jumlah_barang']
    for i in range(df.shape[0]):
        df.iloc[i, -1] = to_timestamp(
            df.iloc[i, 2], df.iloc[i, 1])

    return df


#Data acquisition
def baca_data(produk, tahun):
    db_connection_str = os.getenv('DATABASE_URL')
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


def inferensi(model, x):
    predict = (model.coef_[0][0] * x) + model.intercept_[0]
    return predict


def persamaan_model(model):
    return f"{model.coef_[0][0]} * x + {model.intercept_[0]}"


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


def latih_model(x, y, simpan=True, model_filepath = "model.sav", model = None):
    if model is None:
        model = LinearRegression()
    model.fit(x, y)
    if simpan:
        # db = get_connection()
        # if db:
        #     with db.connect() as con:
        #         con.execute("INSERT INTO model ")
        joblib.dump(model, model_filepath)
    return model


def muat_model(model_filepath):
    err, model = "", None
    if os.path.isfile(model_filepath):
        loaded_model = joblib.load(model_filepath)
        if isinstance(loaded_model, LinearRegression):
            model = loaded_model
        else:
            err = "File Model bukan LinearRegression class"
    else:
        err = "File Model tidak ditemukan. Silahkan latih model."
    return model, err


def muat_model_by_nama_barang(nama_barang):
    barang, err = get_barang_by_nama_produk(nama_barang)
    if err == "":
        print(f"{model_directory()}/{str(barang[0])}_{str(barang[1])}.model")
        model, err = muat_model(
            f"{model_directory()}/{str(barang[0])}_{str(barang[1])}.model")
        if model:
            return model, err
        else:
            err = "Model Barang tidak ditemukan. Silahkan latih ulang model"
    return None, err


def latih_model_satuan(nama_barang, simpan=True):
    error, mape, mse = "", 0, 0
    if nama_barang and nama_barang != "":
        df = ambil_data_barang(nama_barang)
        produk, err = get_barang_by_nama_produk(nama_barang)
        if (err == ""):
            model, err = muat_model_by_nama_barang(produk)
            error = err
            X_train, X_test, y_train, y_test = train_test_split(
                df[['time']], df[['jumlah_barang']], test_size=0.25, random_state=42)
            model = latih_model(
                X_train, y_train, simpan=simpan,
                model_filepath=f"{model_directory()}/{str(produk[0])}_{str(produk[1])}.model",
                model=model)
            y_result = model.predict(X_test)
            mape, mse = evaluasi(y_result, y_test)
    return error, mape, mse


def latih_model_sekaligus():
    df = ambil_data_sekaligus()
    daftar_barang = df['nama_barang'].unique()
    error = []
    for barang in daftar_barang:
        produk, err = get_barang_by_nama_produk(barang)
        if (err == ""):
            dfsub = df.loc[df['nama_barang'] == str(produk[2])]
            dfsub['time'] = dfsub['jumlah_barang']
            for i in range(dfsub.shape[0]):
                dfsub.iloc[i, -1] = to_timestamp(
                    dfsub.iloc[i, 2], dfsub.iloc[i, 1])
            try:
                model, err = muat_model_by_nama_barang(produk)
                model = latih_model(
                    dfsub[['time']], dfsub[['jumlah_barang']], simpan=True,
                    model_filepath=f"{model_directory()}/{str(produk[0])}_{str(produk[1])}.model",
                    model=model)
                time.sleep(1)
            except:
                error.append(f"Terdapat error pada pelatihan model barang {barang}")
    return error


def inferensi_tahunan(model, tahun, pembulatan=True):
    hasil = []
    for i in range(1, 13):
        x = datetime(tahun, i, 1, 0, 0, 0).timestamp()
        prediksi = inferensi(model, x)
        if pembulatan:
            hasil.append(round(prediksi))
        else:
            hasil.append(prediksi)
    return hasil
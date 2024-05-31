from process import * #library yang sudah kita buat
from flask import Flask, request, json


app = Flask(__name__)

#menggunakan method GET karena kita hanya mengambil data bukan memasukkan data ke dalam database
@app.route('/predict', methods=['GET'])
def predict():
  # file_dataset = 'dataset/penjualan_produk.csv'
  produk = str(request.form['nama_produk']) #mengambil data dari form nama produk
  tahun = int(request.form['tahun']) #mengambil data dari form tahun

  #data acquisition
  df = baca_data(produk, tahun)

  #menentukan kolom soal
  x = df[['x']]

  #menentukan kolom jawaban
  y = df['jumlah_barang'].values

  intercept, coeff = model(x, y)

  #memprediksi data test
  predict = (coeff * x) + intercept
  df['prediksi'] = predict

  #evaluasi
  mape, mse = evaluasi(df['jumlah_barang'], df['prediksi'])
  mape = float(round(mape, 2))
  mse = float(round(mse, 2))

  #prediksi data baru
  data_last = df.iloc[-1] #mengambil data terakhir
  data_prediksi = data_last['x'] + 1  #panjang data ditambah dengan 1 (mewakili bulan selanjutnya)
  prediksi_data_baru = (coeff * data_prediksi) + intercept #model least square
  prediksi_data_baru = int(round(prediksi_data_baru[0], 0))

  # info = [data_last[0], str(data_last[2]), str(data_last[3]), prediksi_data_baru, mape, mse]

  data = {'nama_produk':data_last[0], 'tahun':int(data_last[2]), 'bulan':int(data_last[1]),
          'prediksi':prediksi_data_baru, 
          'mape':mape, 'mse':mse}

  #tampung data dan diubah menjadi json
  #json lebih mudah digunakan
  response = app.response_class(
      response=json.dumps(data),
      status=200,
      mimetype='application/json'
  )

  # response = jsonify(data)

  return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
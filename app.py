from process import * #library yang sudah kita buat
from flask import Flask, request, json

app = Flask(__name__)


# REVISI
# Fungsi Latih model sudah dipisah, dan melatih model tiap barang sekaligus.
# Apabila terdapat 100 barang, maka dihasilkan 100 model
@app.route('/bulktraining', methods=['GET'])
def training_model_bulk():
  data = {"success": False, "data": [], "message": None}
  model_evaluation, error = latih_model_sekaligus()
  if error != "":
     if len(error) > 0:
        data['message'] = error
     else:
        data['data'] = model_evaluation
        data['success'] = True
  response = app.response_class(
     response=json.dumps(data),
     status=200,
     mimetype='application/json'
     )
  return response
  # response = jsonify(data)


# @app.route('/each_training', methods=['GET'])
# def training_model_each():
#   produk = str(request.form['nama_produk']) #mengambil data dari form nama produk
#   if produk != "":
#      error, mape, mse = latih_model_satuan(produk)
#   data = {"error": error, "mape": mape, "mse": mse}
#   response = app.response_class(
#      response=json.dumps(data),
#      status=200,
#      mimetype='application/json'
#      )
#   return response
#   # response = jsonify(data)



#menggunakan method GET karena kita hanya mengambil data bukan memasukkan data ke dalam database
@app.route('/predict_old', methods=['GET'])
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

# REVISI
# Fungsi Untuk menampilkan persamaan (rumus) model tiap barang
@app.route('/persamaan', methods=['GET', 'POST'])
def persamaan_model_barang():
   # mengambil data dari form nama produk
   produk = str(request.form['nama_produk'])
   data = {"success": False, "persamaan": "", "message": ""}
   try:
      model, err = muat_model_by_nama_barang(produk)
      if model:
         data['persamaan'] = persamaan_model(model)
         data['success'] = True
      else:
         data['message'] = err
   except Exception as e:
      data['message'] = e.__str__()
   response = app.response_class(
      response=json.dumps(data),
      status=200,
      mimetype='application/json'
      )
   return response


# REVISI
# Fungsi Prediksi sudah dipisah, dan inputan dari web hanya tahun dan nama barang
# Lalu dihasilkan nilai prediksi (total penjualan) untuk tiap bulanannya.
@app.route('/predict', methods=['GET', 'POST'])
def prediksi_tahunan():
   # file_dataset = 'dataset/penjualan_produk.csv'
   # mengambil data dari form nama produk
   produk = str(request.form['nama_produk'])
   tahun = int(request.form['tahun'])  # mengambil data dari form tahun
   prediksi = None
   
   #   Ambil model berdasarkan nama barang
   model, err = muat_model_by_nama_barang(produk)
   if model:
      # menentukan kolom soal
      try:
         prediksi = inferensi_tahunan(model, tahun, nama_barang=produk)
      except Exception as e:
         err = e.__str__()
   else:
      prediksi = None
      err = str("model/rumus regresi linear untuk barang "
                + f"{produk} tidak ditemukan")

   # info = [data_last[0], str(data_last[2]), str(data_last[3]), prediksi_data_baru, mape, mse]

   data = {'nama_produk': produk, 'tahun': int(tahun),
      'bulan': [x for x in range(1, 13)], "message": err,
      'prediksi': prediksi}

   # tampung data dan diubah menjadi json
   # json lebih mudah digunakan
   response = app.response_class(
      response=json.dumps(data),
      status=200,
      mimetype='application/json'
   )

   # response = jsonify(data)

   return response


if __name__ == "__main__":
   app.run(host="0.0.0.0", debug=True)
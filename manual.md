# Linear Regression 

## Rumus/Persamaan Linear Regression
Pada data penjualan yang tersedia, dibuatlah persamaan untuk menentukan/memprediksi
banyaknya jumlah penjualan (output) berdasarkan faktor (inputan) waktu yang tersedia. Faktor waktu pada kasus ini
berupa tahun dan bulan yang nantinya akan diubah ke dalam bentuk integer timestamp (bilangan bulat/angka positif, 
contoh)

Integer timestamp adalah hitungan detik yang dimulai sejak 1 Januari 1970 00:00:00.
1970-01-01 00:00:01 = 1;
1970-01-01 00:00:02 = 2;
1970-01-01 00:00:03 = 3;
1970-01-01 00:01:00 = 60;
1970-01-01 00:02:00 = 120; dan seterusnya hingga detik saat ini

dan dihasilkan persamaan/rumus sebagai berikut.

Jumlah_Penjualan = Koefisien x Waktu + Intersep

Tiap barang menerapkan persamaan/rumus yang sama dengan nilai koefisien dan intersep yang berbeda 
menyesuaikan data latih (data ril) yang telah dikumpulkan.
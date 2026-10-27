# Semboro Orange Quality Classification System 🍊
> **J-TECH Team** | Politeknik Negeri Jember (Polije)  
> Platform Cerdas Klasifikasi Kualitas Jeruk Semboro (*Citrus nobilis var. macrocarpa*) berbasis Deep Learning VGG16 terintegrasi dengan IoT.

---

## 📌 Deskripsi Proyek
Proyek ini mengimplementasikan sistem otomasi untuk menentukan grade mutu jeruk Semboro secara real-time. Memadukan arsitektur Computer Vision **VGG16**, mikrokontroler (Arduino / ESP8266) untuk sistem mekanik sortasi, serta antarmuka monitoring berbasis web.

### 🏷️ Kategori Grade
Model mengklasifikasikan jeruk ke dalam 3 kelas:
1. **Grade A**: Jeruk kualitas unggul, ukuran seragam, minim cacat.
2. **Grade B**: Jeruk kualitas sedang / standar konsumsi pasar umum.
3. **Reject**: Jeruk cacat, bercak penyakit, atau tidak layak pasar segar.

---

## ⚙️ Arsitektur & Alur Kerja
1. **Pendeteksian Objek (Vision)**: Kamera menangkap frame, melakukan segmentasi warna HSV dan kontur jeruk.
2. **Preprocessing**: Citra dipotong (crop) dan dinormalisasi sesuai input standar VGG16 (224x224 px).
3. **Inferensi Model**: Model deep learning VGG16 memprediksi kelas probabilitas tertinggi.
4. **IoT Bridge**: Hasil klasifikasi dikirimkan melalui komunikasi Serial / Telemetri ke aktuator mikrokontroler untuk menggerakkan mekanisme sortir fisik.
5. **Monitoring Web**: Status dan metrik klasifikasi dimonitor secara real-time.

---

## 📂 Struktur Repositori
```plaintext
project_jtech/
├── models/
│   ├── vgg16_jeruk.h5          # Model VGG16 hasil pelatihan (Git LFS)
│   └── vgg16_finetuned.h5      # Model fine-tuned teroptimasi (Git LFS)
├── outputs/
│   ├── batch_predictions.csv   # Hasil evaluasi batch
│   ├── predict.txt             # Log output inferensi
│   └── predict_overlay.jpg     # Contoh visualisasi prediksi dengan bounding box
├── reports/
│   ├── classification_report.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.png    # Matriks konfusi performa model
│   ├── history.csv             # Riwayat metrik training & validation
│   └── history.png             # Kurva loss & akurasi
├── batch_predict.py            # Script prediksi folder massal
├── evaluate.py                 # Script pengujian metrik model
├── fine_tune.py                # Script fine-tuning model VGG16
├── iot_bridge.py               # Sistem sortasi otomatis terhubung webcam & Arduino
├── main.py                     # Pipeline training, augmentasi data, dan evaluasi
├── app.py                      # Server monitoring berbasis Flask
├── requirements.txt            # Dependensi Python
└── README.md
```

---

## 🚀 Panduan Memulai

### 1. Prasyarat & Kloning Repositori
Pastikan Anda telah menginstal **Python 3.9 - 3.11** dan **Git LFS**:
```bash
# Kloning repositori dengan LFS
git clone https://github.com/rdanuartaa/project_jtech.git
cd project_jtech

# Tarik file model bobot besar jika diperlukan
git lfs pull
```

### 2. Instalasi Dependensi
```bash
# Buat dan aktifkan virtual environment (opsional)
python -m venv .venv
# Pada Windows:
.venv\Scripts\activate
# Pada Linux / macOS:
source .venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

### 3. Menjalankan IoT Sortasi Real-Time
Hubungkan mikrokontroler (Arduino / ESP8266) dan kamera:
```bash
python iot_bridge.py --port COM3 --cam 0 --weights models/vgg16_jeruk.h5
```
*Parameter:*
- `--port`: Port serial (misal `COM3` atau `/dev/ttyUSB0`)
- `--cam`: ID kamera (default `0`)
- `--weights`: Path file model `.h5`

### 4. Menjalankan Training Model
```bash
python main.py
```

### 5. Menjalankan Monitoring Web
```bash
python app.py
```

---

## 👥 Tim Pengembang (J-TECH)
- **Rio Wildan Rakhsanjani A** (E41232146) — *Ketua Kelompok*
- **Muhammad Rafi Syahbana** (E41232076)
- **Rizal Danuarta Akbar** (E41232109)
- **Rahmadani Bimasaka Anggara** (E41232291)
- **Dewitha Josephine Sembiring** (E41232065)

**Program Studi**: Teknik Informatika, Politeknik Negeri Jember  
**Pameran JTI Polije**: [Klasifikasi Kualitas Jeruk Semboro](https://pameran-jti.polije.ac.id/pameran/product/496/klasifikasi-kualitas-jeruk-semboro-menggunakan-arsitektur-vgg16-yang-terintegrasi-dengan-iot-berbasis-web)

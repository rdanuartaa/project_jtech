import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input  # ✅ tambahkan ini


def predict_folder(folder_path, weights_path, img_size=224):
    # Pastikan folder valid
    assert os.path.exists(folder_path), f"Folder tidak ditemukan: {folder_path}"
    assert os.path.exists(weights_path), f"Model tidak ditemukan: {weights_path}"

    # Load model
    print(f"🔹 Memuat model dari: {weights_path}")
    model = load_model(weights_path)

    # Ambil semua file gambar di folder
    img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"🔹 Jumlah gambar yang ditemukan: {len(img_files)}\n")

    results = []
    for fname in tqdm(img_files, desc="🚀 Memproses gambar"):
        img_path = os.path.join(folder_path, fname)
        img = image.load_img(img_path, target_size=(img_size, img_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # ✅ gunakan normalisasi yang sama dengan VGG16

        preds = model.predict(x, verbose=0)[0]
        classes = ['grade A', 'grade B', 'reject']
        pred_idx = np.argmax(preds)
        pred_class = classes[pred_idx]
        confidence = preds[pred_idx] * 100

        results.append({
            'filename': fname,
            'predicted_class': pred_class,
            'confidence': round(confidence, 2),
            'grade A %': round(preds[0] * 100, 2),
            'grade B %': round(preds[1] * 100, 2),
            'reject %': round(preds[2] * 100, 2)
        })

    # Simpan hasil ke file CSV
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "batch_predictions.csv")
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\n✅ Hasil prediksi disimpan di: {output_path}")


if __name__ == "__main__":
    folder = "test_images"  # ganti sesuai nama folder kamu
    weights = "models/vgg16_jeruk.h5"
    predict_folder(folder, weights)

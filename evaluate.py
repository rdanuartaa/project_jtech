import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# Konfigurasi dasar
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vgg16_finetuned.h5")
   # ubah ke .keras jika kamu pakai format baru
TEST_DIR = os.path.join(BASE_DIR, "data_split", "test")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# Pastikan folder report ada
os.makedirs(REPORT_DIR, exist_ok=True)

# ===============================
# Load model
# ===============================
print("🔹 Memuat model terbaik dari:", MODEL_PATH)
model = load_model(MODEL_PATH)

# ===============================
# Siapkan data test
# ===============================
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# ===============================
# Evaluasi model
# ===============================
print("\n🔍 Mengevaluasi model...")
loss, accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy : {accuracy:.4f}")
print(f"✅ Test Loss     : {loss:.4f}")

# ===============================
# Prediksi dan laporan hasil
# ===============================
print("\n📊 Membuat prediksi...")
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Nama kelas
class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(test_generator.classes, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix - VGG16 Jeruk")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Simpan gambar confusion matrix
cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
print(f"🖼️ Confusion matrix disimpan di: {cm_path}")
plt.close()

# Classification Report
report = classification_report(test_generator.classes, y_pred, target_names=class_labels)
report_path = os.path.join(REPORT_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"📝 Classification report disimpan di: {report_path}")

# ===============================
# Visualisasi Akurasi & Loss (jika file history ada)
# ===============================
history_path = os.path.join(REPORT_DIR, "training_history.npy")
if os.path.exists(history_path):
    print("\n📈 Menampilkan grafik training history...")
    history = np.load(history_path, allow_pickle=True).item()

    plt.figure(figsize=(8, 4))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "accuracy_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "loss_curve.png"))
    plt.close()

    print("📉 Grafik accuracy & loss disimpan di folder reports/")
else:
    print("⚠️ File history tidak ditemukan, lewati visualisasi training curve.")

print("\n✅ Evaluasi selesai. Semua hasil disimpan di folder 'reports'.")

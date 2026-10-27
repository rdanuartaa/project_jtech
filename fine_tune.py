import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.vgg16 import preprocess_input


# =============================
# 1. Fungsi membuat generator
# =============================
def make_gens(data_dir, img_size=224, batch_size=16):
    """
    Membuat generator train & validasi dengan augmentasi data
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen


# =============================
# 2. Hitung class weight
# =============================
def compute_weights(train_gen):
    """
    Hitung class weight otomatis berdasarkan distribusi kelas
    """
    classes = np.array(train_gen.classes)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    return dict(enumerate(class_weights))


# =============================
# 3. Fine-tuning model
# =============================
def fine_tune(model_path, train_gen, val_gen,
              unfreeze_last_n=8, lr=1e-5, epochs=30,
              out_path="models/vgg16_finetuned.h5"):

    # Load model lama
    model = load_model(model_path)
    print(f"✅ Model berhasil dimuat dari: {model_path}")

    # Bekukan semua layer kecuali beberapa terakhir
    for layer in model.layers[:-unfreeze_last_n]:
        layer.trainable = False
    for layer in model.layers[-unfreeze_last_n:]:
        layer.trainable = True

    # Kompilasi ulang model dengan learning rate kecil
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callback untuk menyimpan hasil terbaik
    callbacks = [
        ModelCheckpoint(out_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # Hitung bobot kelas
    class_weights = compute_weights(train_gen)
    print("📊 Class Weights:", class_weights)

    # Mulai training
    hist = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )

    print(f"💾 Model fine-tuned disimpan ke: {out_path}")
    return model, hist


# =============================
# 4. Eksekusi langsung (opsional)
# =============================
if __name__ == "__main__":
    data_dir = "data_split"              # sesuaikan dengan direktori data kamu
    base_model_path = "models/vgg16_jeruk.h5"  # model awal hasil training pertama
    output_model_path = "models/vgg16_finetuned.h5"

    # Buat generator
    train_gen, val_gen = make_gens(data_dir)

    # Fine-tuning
    fine_tune(base_model_path, train_gen, val_gen, unfreeze_last_n=8,
              lr=1e-5, epochs=30, out_path=output_model_path)

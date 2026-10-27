import os
import shutil
import argparse
import random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# =============== Util umum ===============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
REPORTS_DIR = HERE / "reports"
OUTPUTS_DIR = HERE / "outputs"
for d in [MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def imshow_rgb(img_bgr, title=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()

# =============== 1) Split dataset ===============

def split_dataset(src_dir, dst_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    src = Path(src_dir)
    dst = Path(dst_dir)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Rasio harus total 1.0"

    for split in ["train", "val", "test"]:
        (dst / split).mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in src.iterdir() if d.is_dir()]
    print("Kelas terdeteksi:", classes)

    valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    for cls in classes:
        files = [p for p in (src / cls).glob("*.*") if p.suffix.lower() in valid_ext]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            out_dir = dst / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out_dir / f.name)
        print(f"{cls}: train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")

# =============== 2) DataLoader ===============

def make_gens(data_dir, img_size=224, batch_size=16):
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    test_dir = Path(data_dir) / "test"

    class_list = ['grade A', 'grade B', 'reject']  # urutan tetap

    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.25,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )
    valid_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_aug.flow_from_directory(
        train_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', shuffle=True, seed=SEED,
        classes=class_list
    )
    val_gen = valid_aug.flow_from_directory(
        val_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', shuffle=False,
        classes=class_list
    )
    test_gen = valid_aug.flow_from_directory(
        test_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', shuffle=False,
        classes=class_list
    )

    print("Class indices:", train_gen.class_indices)
    return train_gen, val_gen, test_gen




# =============== 3) Bangun model VGG16 ===============

def build_model(num_classes, img_size=224, freeze_until=15):
    base = VGG16(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
    # Bekukan sebagian layer untuk transfer learning stabil
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= freeze_until)

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base.input, outputs=out)
    opt = optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =============== 4) Train ===============

from sklearn.utils import class_weight
import numpy as np

def train_model(data_dir, img_size=224, batch=16, epochs=30, freeze=15):
    # Load generator
    train_gen, val_gen, _ = make_gens(data_dir, img_size, batch)

    # Bangun model
    model = build_model(num_classes=train_gen.num_classes, img_size=img_size, freeze_until=freeze)

    # ✅ Hitung class weight di dalam fungsi, setelah train_gen didefinisikan
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("🔹 Class Weights:", class_weights)

    # Callback
    ckpt_path = MODELS_DIR / 'vgg16_finetuned.h5'
    callbacks = [
        ModelCheckpoint(str(ckpt_path), monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(patience=6, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1, monitor='val_loss')
    ]

    # Training
    hist = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    return model, hist




    # Simpan grafik training
    hist_df = pd.DataFrame(hist.history)
    hist_csv = REPORTS_DIR / 'history.csv'
    hist_png = REPORTS_DIR / 'history.png'
    hist_df.to_csv(hist_csv, index=False)
    plt.figure(figsize=(7, 4))
    plt.plot(hist.history['accuracy'], label='train_acc')
    plt.plot(hist.history['val_accuracy'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
    plt.savefig(hist_png, dpi=150)
    print(f"History tersimpan: {hist_csv}\nPlot: {hist_png}")

# =============== 5) Evaluasi ===============

def evaluate_model(data_dir, weights, img_size=224, batch=16):
    _, _, test_gen = make_gens(data_dir, img_size, batch)
    model = build_model(num_classes=test_gen.num_classes, img_size=img_size)
    model.load_weights(weights)

    loss, acc = model.evaluate(test_gen)
    print(f"Test Loss={loss:.4f} | Test Acc={acc:.4f}")

    y_true = test_gen.classes
    y_prob = model.predict(test_gen)
    y_pred = np.argmax(y_prob, axis=1)

    labels = list(test_gen.class_indices.keys())
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    # Simpan laporan
    cm_png = REPORTS_DIR / 'confusion_matrix.png'
    cr_csv = REPORTS_DIR / 'classification_report.csv'

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Prediksi'); plt.ylabel('Aktual'); plt.tight_layout(); plt.savefig(cm_png, dpi=150)

    pd.DataFrame(cr).to_csv(cr_csv)
    print(f"Saved: {cm_png}, {cr_csv}")

# =============== 6) Prediksi tunggal ===============

def load_model_with_weights(weights, num_classes=3, img_size=224):
    model = build_model(num_classes=num_classes, img_size=img_size)
    model.load_weights(weights)
    return model


def predict_image(img_path, weights, class_names=None, img_size=224):
    assert os.path.exists(img_path), f"Gambar tidak ditemukan: {img_path}"

    if class_names is None:
        # default urutan folder train jika ada
        class_names = ['grade A', 'grade B', 'reject']

    model = load_model_with_weights(weights, num_classes=len(class_names), img_size=img_size)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"Gagal membaca gambar: {img_path}")

    img_resized = cv2.resize(img_bgr, (img_size, img_size))
    x = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    prob = model.predict(x)[0]
    top = int(np.argmax(prob))
    label = class_names[top]
    conf = prob[top]

    out_txt = OUTPUTS_DIR / 'predict.txt'
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f"Prediksi: {label} ({conf:.2%})\n")
        for i, cname in enumerate(class_names):
            f.write(f"- {cname}: {prob[i]:.2%}\n")

    print(open(out_txt, 'r', encoding='utf-8').read())

    # Simpan overlay teks pada gambar
    out_img = OUTPUTS_DIR / 'predict_overlay.jpg'
    vis = img_bgr.copy()
    cv2.putText(vis, f"{label} {conf:.1%}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_img), vis)
    print(f"Gambar hasil: {out_img}")

# =============== 7) Grad-CAM (deteksi area cacat) ===============

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='block5_conv3'):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-8)
    return heatmap.numpy()


def gradcam(img_path, weights, img_size=224, alpha=0.35):
    model = load_model_with_weights(weights, num_classes=3, img_size=img_size)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"Gagal membaca gambar: {img_path}")
    img_resized = cv2.resize(img_bgr, (img_size, img_size))

    x = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    heatmap = make_gradcam_heatmap(x, model)
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)

    out_path = OUTPUTS_DIR / 'gradcam_overlay.jpg'
    cv2.imwrite(str(out_path), overlay)
    print(f"Grad-CAM tersimpan: {out_path}")

# =============== 8) Export TFLite ===============

def export_tflite(weights, img_size=224):
    model = load_model_with_weights(weights, num_classes=3, img_size=img_size)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    out_file = MODELS_DIR / 'vgg16_jeruk.tflite'
    with open(out_file, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite tersimpan: {out_file}")

# =============== CLI ===============

def main():
    parser = argparse.ArgumentParser(description='Sistem Sortasi Jeruk – VGG16')
    sub = parser.add_subparsers(dest='command')

    p_split = sub.add_parser('split')
    p_split.add_argument('--src', required=True)
    p_split.add_argument('--dst', default='data_split')
    p_split.add_argument('--train', type=float, default=0.7)
    p_split.add_argument('--val', type=float, default=0.15)
    p_split.add_argument('--test', type=float, default=0.15)

    p_train = sub.add_parser('train')
    p_train.add_argument('--data', default='data_split')
    p_train.add_argument('--img-size', type=int, default=224)
    p_train.add_argument('--batch', type=int, default=16)
    p_train.add_argument('--epochs', type=int, default=30)
    p_train.add_argument('--freeze', type=int, default=15, help='bekukan layer < freeze')

    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--data', default='data_split')
    p_eval.add_argument('--img-size', type=int, default=224)
    p_eval.add_argument('--batch', type=int, default=16)
    p_eval.add_argument('--weights', required=True)

    p_pred = sub.add_parser('predict')
    p_pred.add_argument('--img', required=True)
    p_pred.add_argument('--weights', required=True)
    p_pred.add_argument('--img-size', type=int, default=224)

    p_gc = sub.add_parser('gradcam')
    p_gc.add_argument('--img', required=True)
    p_gc.add_argument('--weights', required=True)
    p_gc.add_argument('--img-size', type=int, default=224)

    p_tfl = sub.add_parser('export-tflite')
    p_tfl.add_argument('--weights', required=True)
    p_tfl.add_argument('--img-size', type=int, default=224)

    args = parser.parse_args()

    if args.command == 'split':
        split_dataset(args.src, args.dst, args.train, args.val, args.test)
    elif args.command == 'train':
        train_model(args.data, args.img_size, args.batch, args.epochs, args.freeze)
    elif args.command == 'eval':
        evaluate_model(args.data, args.weights, args.img_size, args.batch)
    elif args.command == 'predict':
        predict_image(args.img, args.weights, img_size=args.img_size)
    elif args.command == 'gradcam':
        gradcam(args.img, args.weights, img_size=args.img_size)
    elif args.command == 'export-tflite':
        export_tflite(args.weights, args.img_size)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

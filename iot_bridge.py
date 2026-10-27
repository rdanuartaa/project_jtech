# iot_bridge.py (versi otomatis - vision only)
import cv2, numpy as np, serial, time, argparse
from pathlib import Path
import main as jeruk  # model CNN dari main.py

CLASS_NAMES = ['grade A', 'grade B', 'reject']

def preprocess_for_vgg(img_bgr, img_size=224):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = cv2.resize(rgb, (img_size, img_size))
    x = np.expand_dims(rgb, axis=0)
    x = jeruk.preprocess_input(x)
    return x

def decide_label(probs):
    idx = int(np.argmax(probs))
    return ['A', 'B', 'R'][idx], CLASS_NAMES[idx], float(probs[idx])

def detect_orange(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 100, 100])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 10000:  # batas minimal luas jeruk (sesuaikan)
        return None, mask
    (x, y, w, h) = cv2.boundingRect(largest)
    return (x, y, w, h), mask

def run(port, baud, weights, cam_id, img_size):
    print(f"Hubungkan ke Arduino di {port}...")
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(2)

    print("Memuat model CNN...")
    model = jeruk.load_model_with_weights(weights, num_classes=len(CLASS_NAMES), img_size=img_size)

    cap = cv2.VideoCapture(cam_id)
    assert cap.isOpened(), "Kamera tidak terdeteksi"

    print("=== SISTEM SORTASI OTOMATIS AKTIF ===")
    print("Letakkan jeruk di bawah kamera...")

    last_detect_time = 0
    cooldown = 3  # detik, jeda antar deteksi

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        box, mask = detect_orange(frame)
        display = frame.copy()
        if box:
            x, y, w, h = box
            cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(display, "Jeruk Terdeteksi", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # hanya klasifikasi jika cooldown selesai
            if time.time() - last_detect_time > cooldown:
                crop = frame[y:y+h, x:x+w]
                x_input = preprocess_for_vgg(crop, img_size)
                probs = model.predict(x_input, verbose=0)[0]
                cmd, label, conf = decide_label(probs)

                ser.write((cmd + '\n').encode('utf-8'))
                print(f"Prediksi: {label} ({conf:.1%}) → Kirim '{cmd}' ke Arduino")

                cv2.putText(display, f"{label} ({conf:.1%})", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                last_detect_time = time.time()

        cv2.imshow("Sortasi Jeruk Otomatis", display)
        cv2.imshow("Mask Warna Jeruk", mask)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
            break

    cap.release()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vision-only CNN Sorting (tanpa IR sensor)")
    ap.add_argument("--port", required=True, help="misalnya COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--weights", default=str(Path("models")/"vgg16_jeruk.h5"))
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()
    run(args.port, args.baud, args.weights, args.cam, args.img_size)

import os
import sys
import time
import threading
import json
import base64

gst_bin_path = r'D:\program\msvc_x86_64\bin'

if os.path.exists(gst_bin_path):
    os.add_dll_directory(gst_bin_path)

import cv2
import redis


def camera_worker(cam_id, video_file):
    # Убираем проверку GStreamer DLL для этого теста, используем стандартный OpenCV
    r = redis.Redis(host='localhost', port=6379)

    if not os.path.exists(video_file):
        print(f"[Ошибка] Файл не найден: {video_file}")
        return

    # Открываем БЕЗ GStreamer (просто путь к файлу)
    # Это в разы стабильнее для 32 камер с диска
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"[Ошибка] OpenCV не смог открыть файл для Cam {cam_id}")
        return

    print(f"[Cam {cam_id}] Успешно запущен (Standart Backend)")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Ресайз до 640x640 (так как мы убрали GStreamer videoscale)
        frame_resized = cv2.resize(frame, (640, 640))

        # Кодируем в JPEG
        _, img_encoded = cv2.imencode('.jpg', frame_resized)

        payload = {
            "cam_id": cam_id,
            "img": base64.b64encode(img_encoded.tobytes()).decode("ascii"),
        }
        r.lpush("image_batch_queue", json.dumps(payload))
        r.ltrim("image_batch_queue", 0, 1000)

        # Контроль FPS (25 кадров)
        time.sleep(0.2)


if __name__ == "__main__":
    folder_num = int(sys.argv[1])  # Например, 1, 2, 3 или 4

    threads = []
    for i in range(1, 9):
        # Вычисляем уникальный ID для Redis
        # Если folder_num = 1: 1..8
        # Если folder_num = 2: 9..16
        # Если folder_num = 3: 17..24
        # Если folder_num = 4: 25..32
        unique_id = (folder_num - 1) * 8 + i

        # Путь к файлу (тут оставляем i, если внутри папок файлы называются test1..test8)
        video_file = os.path.join(f"video{folder_num}", f"test{i}.mp4")

        # Передаем unique_id как cam_id
        t = threading.Thread(target=camera_worker, args=(str(unique_id), video_file))
        t.daemon = True
        t.start()
        threads.append(t)
        time.sleep(0.5)

    print(f"Запущено {len(threads)} камер. Нажми Ctrl+C для выхода.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Остановка...")
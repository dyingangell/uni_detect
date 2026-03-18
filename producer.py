import redis
import sys
import os

# Добавляем путь к твоим DLL GStreamer (так как это сработало в тесте)
gst_bin_path = r'D:\program\msvc_x86_64\bin'
if os.path.exists(gst_bin_path):
    os.add_dll_directory(gst_bin_path)
import cv2

def start_producer(cam_id, source):
    r = redis.Redis(host='localhost', port=6379)

    # 1. Автоматически выбираем источник
    if source.startswith("rtsp://"):
        # Пайплайн для RTSP (с защитой от задержек)
        gst_source = f'rtspsrc location={source} latency=0 protocols=tcp ! rtph264depay ! h264parse ! decodebin'
    else:
        # Пайплайн для локальных файлов (типа test7.mp4)
        # filesrc читает файл, decodebin сам поймет формат
        gst_source = f'filesrc location="{source}" ! decodebin'

    # 2. Собираем финальный пайплайн (универсальная часть для обработки)
    gst_pipeline = (
        f'{gst_source} ! '
        f'videoscale ! '
        f'video/x-raw, width=640, height=640 ! '
        f'videoconvert ! '
        f'appsink drop=True'
    )

    print(f"Запуск пайплайна: {gst_pipeline}")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"Ошибка: Не удалось запустить GStreamer для камеры {cam_id}")
        return

    print(f"Producer запущен для камеры {cam_id} через GStreamer...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Потеряно соединение с камерой {cam_id}. Переподключение...")
            cap.release()
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            continue

        # Записываем в Redis
        # frame уже 640x640 благодаря GStreamer pipeline, resize() в Python больше не нужен!
        _, img_encoded = cv2.imencode('.jpg', frame)
        r.lpush("image_batch_queue", img_encoded.tobytes()) # Тут должен быть img_encoded!
        r.ltrim("image_batch_queue", 0, 100)
        print(f"[{cam_id}] Frame sent to Redis") # Добавь для проверки

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Использование: python producer.py <id> <rtsp_url>")
    else:
        start_producer(sys.argv[1], sys.argv[2])
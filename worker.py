import os
import time
from concurrent.futures import ThreadPoolExecutor

import redis
import numpy as np
gst_bin_path = r'D:\program\msvc_x86_64\bin'
if os.path.exists(gst_bin_path):
    os.add_dll_directory(gst_bin_path)
import cv2

from newArch import ProctoringEngine

r = redis.Redis(host='localhost', port=6379)
keys = r.keys("camera:*")
engine =  ProctoringEngine()
r.flushall()
print("Worker started. Processing batches...")
def decode_img(img_bytes):
    if not img_bytes:
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
executor = ThreadPoolExecutor(max_workers=8)
while True:
    start_time = time.time()

    # 1. Быстро собираем байты из Redis и СРАЗУ удаляем их (LIFO/Zero-latency)
    raw_data = []
    active_cam_ids = []

    for key in keys:
        img_bytes = r.get(key)
        if img_bytes:
            r.delete(key) # Гарантируем отсутствие очереди
            raw_data.append(img_bytes)
            active_cam_ids.append(key)

    if not raw_data:
        continue # Ждем данные, если пусто

    # 2. Параллельное декодирование (CPU Multithreading)
    # Это "схлопывает" время декодирования 16-64 камер
    frames_for_engine = list(executor.map(decode_img, raw_data))

    # Очищаем от None (если imdecode не справился)
    frames_for_engine = [f for f in frames_for_engine if f is not None]

    if frames_for_engine:
        # 3. Пакетная обработка (GPU Inference)
        processed_frames, detections = engine.process_batch(frames_for_engine)

        # 4. Сохранение результатов (JPEG encoding можно тоже в ThreadPool, если будет тормозить)
        # for i, cam_id in enumerate(active_cam_ids):
        #     # Ставим качество 60-80 для экономии ресурсов
        #     _, buffer = cv2.imencode('.jpg', processed_frames[i], [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        #     r.set(f"result:{cam_id}", buffer.tobytes())

        # Статистика
        total_time = time.time() - start_time
        fps = len(frames_for_engine) / total_time
        latency_ms = total_time * 1000

        print(f"Batch: {len(frames_for_engine)} | System FPS: {fps:.2f} | Latency: {latency_ms:.1f}ms   ", end='\r')
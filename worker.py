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
engine =  ProctoringEngine()
r.flushall()
print("Worker started. Processing batches...")
def decode_img(img_bytes):
    if not img_bytes:
        return None
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
executor = ThreadPoolExecutor(max_workers=8)
QUEUE_NAME = "image_batch_queue" # Договорись с продюсерами, чтобы они делали r.lpush(QUEUE_NAME, img_bytes)
MAX_BATCH_SIZE = 64

while True:
    t_start = time.time()

    # --- Шаг 1: Забор из Redis ---
    pipe = r.pipeline()
    pipe.lrange(QUEUE_NAME, 0, -1)
    pipe.delete(QUEUE_NAME)
    results = pipe.execute()

    all_data = results[0]
    t_after_redis = time.time()

    if not all_data:
        time.sleep(0.005)
        continue

    # Берем срез, чтобы не перегружать CPU (можешь менять 64 на 16 или 32 для теста)
    raw_data = all_data[:64]

    # --- Шаг 2: Декодирование картинок (CPU) ---
    frames_for_engine = list(executor.map(decode_img, raw_data))
    frames_for_engine = [f for f in frames_for_engine if f is not None]
    t_after_decode = time.time()

    if not frames_for_engine:
        continue

    # --- Шаг 3: Нейронка (GPU Inference + Postprocess) ---
    # ВАЖНО: убедись, что внутри process_batch нет лишних time.sleep или принтов
    processed_frames, detections = engine.process_batch(frames_for_engine)
    t_after_gpu = time.time()

    # --- Расчет таймингов ---
    total_time = t_after_gpu - t_start

    # Сколько мс занял каждый этап:
    redis_ms = (t_after_redis - t_start) * 1000
    decode_ms = (t_after_decode - t_after_redis) * 1000
    gpu_ms = (t_after_gpu - t_after_decode) * 1000
    total_ms = total_time * 1000

    fps = len(frames_for_engine) / total_time

    # Вывод подробной статистики
    stats = (
        f"Batch: {len(frames_for_engine)} | "
        f"Redis: {redis_ms:4.1f}ms | "
        f"Decode: {decode_ms:4.1f}ms | "
        f"GPU: {gpu_ms:4.1f}ms | "
        f"Total: {total_ms:5.1f}ms | "
        f"FPS: {fps:5.2f}"
    )
    print(stats, end='\r')
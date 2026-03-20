import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import base64

import redis
import numpy as np
import multiprocessing as mp

gst_bin_path = r'D:\program\msvc_x86_64\bin'
if os.path.exists(gst_bin_path):
    os.add_dll_directory(gst_bin_path)
import cv2

from newArch import ProctoringEngine


print("Worker started. Processing batches...")

# settings
QUEUE_NAME = "image_batch_queue"  # Продюсеры делают r.lpush(QUEUE_NAME, json)
MAX_BATCH_SIZE = 16


def decode_img(item_bytes):
    """Декодируем один элемент очереди: поддерживаем и JSON, и сырые JPEG-байты на всякий случай."""
    if not item_bytes:
        return None

    # Если пришёл старый формат (сырые JPEG-байты)
    if len(item_bytes) >= 2 and item_bytes[0] == 0xFF and item_bytes[1] == 0xD8:
        nparr = np.frombuffer(item_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return (frame, "unknown")

    # Новый формат: JSON {"cam_id": "...", "img": "<base64 jpeg>"}
    if isinstance(item_bytes, bytes):
        item_bytes = item_bytes.decode("utf-8", errors="strict")

    data = json.loads(item_bytes)
    cam_id = str(data["cam_id"])
    img_bytes = base64.b64decode(data["img"])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return (frame, cam_id)


def ai_process_worker(input_queue):
    # Модель инициализируется ВНУТРИ процесса для доступа к CUDA
    engine = ProctoringEngine()
    print("AI Engine Process started and model loaded.")

    while True:
        data = input_queue.get()
        if data is None:
            break  # Сигнал остановки
        frames, cam_ids = data

        t_gpu_start = time.time()
        # Сама обработка
        processed_frames, detections = engine.process_batch(frames, cam_ids)

        gpu_ms = (time.time() - t_gpu_start) * 1000
        print(f"[GPU] Processed batch of {len(frames)} in {gpu_ms:.1f}ms")
def results_sender(results_queue, redis_host='localhost'):
    """Фоновый поток для отправки данных в Redis"""
    import redis
    r = redis.Redis(host=redis_host, port=6379)
    print("[Sender] Async sender started.")

    while True:
        data = results_queue.get()
        if data is None: break

        # Используем pipeline для массовой отправки, если батч большой
        pipe = r.pipeline()
        for item in data:
            pipe.rpush("visualize_queue", json.dumps(item))
        pipe.execute()
# --- ОСНОВНОЙ МЕНЕДЖЕР (DECODER) ---
if __name__ == "__main__":
    r = redis.Redis(host='localhost', port=6379)
    # Дружелюбная проверка Redis вместо падения трассировкой
    try:
        r.ping()
    except Exception as e:
        print("[Ошибка] Redis недоступен на localhost:6379.")
        print("Запусти Redis и повтори. Если используешь Docker Desktop, пример:")
        print('  docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest')
        print(f"Детали: {e}")
        raise SystemExit(2)

    # очистка очередей/состояния (безопаснее чем FLUSHALL)
    try:
        r.delete(QUEUE_NAME, "raw_ai_results", "proctor_warnings")
    except Exception:
        pass
    executor = ThreadPoolExecutor(max_workers=8)

    # Очередь для передачи кадров в ИИ (храним не более 3 батчей, чтобы не забить RAM)
    frames_queue = mp.Queue(maxsize=3)

    # Запускаем процесс ИИ
    p = mp.Process(target=ai_process_worker, args=(frames_queue,))
    p.start()

    print("Worker Manager started. Decoding and feeding AI...")

    try:
        while True:
            t_start = time.time()

            # 1. Забор из Redis (Pipeline ускоряет работу)
            pipe = r.pipeline()
            pipe.lrange(QUEUE_NAME, 0, MAX_BATCH_SIZE - 1)
            pipe.ltrim(QUEUE_NAME, MAX_BATCH_SIZE, -1)
            results = pipe.execute()
            raw_data = results[0]
            t_after_redis = time.time()
            if not raw_data:
                time.sleep(0.01)
                continue

            # 2. Декодирование картинок (Параллельно на CPU)
            items = list(executor.map(decode_img, raw_data))
            items = [it for it in items if it is not None]
            frames = [it[0] for it in items]
            cam_ids = [it[1] for it in items]
            t_after_decode = time.time()
            if frames:
                # 3. Отправляем в очередь ИИ БЕЗ ОЖИДАНИЯ результата
                # Если очередь полна, метод заблокируется, пока ИИ не освободится
                frames_queue.put((frames, cam_ids))
            t_after_gpu = time.time()
            decode_ms = (time.time() - t_start) * 1000
            fps_input = len(frames) / (time.time() - t_start)

            #     # --- Расчет таймингов ---
            total_time = t_after_gpu - t_start

            # Сколько мс занял каждый этап:
            redis_ms = (t_after_redis - t_start) * 1000
            decode_ms = (t_after_decode - t_after_redis) * 1000
            gpu_ms = (t_after_gpu - t_after_decode) * 1000
            total_ms = total_time * 1000

            fps = len(frames) / total_time

            # Вывод подробной статистики
            stats = (
                f"Batch: {len(frames)} | "
                f"Redis: {redis_ms:4.1f}ms | "
                f"Decode: {decode_ms:4.1f}ms | "
                f"GPU: {gpu_ms:4.1f}ms | "
                f"Total: {total_ms:5.1f}ms | "
                f"FPS: {fps:5.2f}"
            )
            print(stats)

    except KeyboardInterrupt:
        frames_queue.put(None)
        p.join()
# while True:
#     t_start = time.time()
#
#     # --- Шаг 1: Забор из Redis ---
#     pipe = r.pipeline()
#     pipe.lrange(QUEUE_NAME, 0, -1)
#     pipe.delete(QUEUE_NAME)
#     results = pipe.execute()
#
#     all_data = results[0]
#     t_after_redis = time.time()
#
#     if not all_data:
#         time.sleep(0.005)
#         continue
#
#     # Берем срез, чтобы не перегружать CPU (можешь менять 64 на 16 или 32 для теста)
#     raw_data = all_data[:MAX_BATCH_SIZE]
#
#     # --- Шаг 2: Декодирование картинок (CPU) ---
#     frames_for_engine = list(executor.map(decode_img, raw_data))
#     frames_for_engine = [f for f in frames_for_engine if f is not None]
#     t_after_decode = time.time()
#
#     if not frames_for_engine:
#         continue
#
#     # --- Шаг 3: Нейронка (GPU Inference + Postprocess) ---
#     # ВАЖНО: убедись, что внутри process_batch нет лишних time.sleep или принтов
#     processed_frames, detections = engine.process_batch(frames_for_engine)
#     t_after_gpu = time.time()
#
#     # --- Расчет таймингов ---
#     total_time = t_after_gpu - t_start
#
#     # Сколько мс занял каждый этап:
#     redis_ms = (t_after_redis - t_start) * 1000
#     decode_ms = (t_after_decode - t_after_redis) * 1000
#     gpu_ms = (t_after_gpu - t_after_decode) * 1000
#     total_ms = total_time * 1000
#
#     fps = len(frames_for_engine) / total_time
#
#     # Вывод подробной статистики
#     stats = (
#         f"Batch: {len(frames_for_engine)} | "
#         f"Redis: {redis_ms:4.1f}ms | "
#         f"Decode: {decode_ms:4.1f}ms | "
#         f"GPU: {gpu_ms:4.1f}ms | "
#         f"Total: {total_ms:5.1f}ms | "
#         f"FPS: {fps:5.2f}"
#     )
#     print(stats)
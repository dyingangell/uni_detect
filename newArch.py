import json
import os

import numpy as np
import msgpack

gst_bin_path = r'D:\program\msvc_x86_64\bin'
import redis
if os.path.exists(gst_bin_path):
    os.add_dll_directory(gst_bin_path)
import cv2
import time
from ultralytics import YOLO
from multiprocessing import shared_memory
# model_det = YOLO('yolo11m.pt') # или твой путь к .pt
# model_det.export(format='engine', device=0, imgsz=960, half=True, batch=8, dynamic=True)
#
## Для модели поз (Pose) — ОБЯЗАТЕЛЬНО тоже в .engine!
# model_pose = YOLO('yolo11l-pose.pt')
# model_pose.export(format='engine', device=0, imgsz=640, half=True, batch=16, dynamic=True)
MAX_BATCH = 32

class ProctoringEngine:
    def __init__(self, pose_path='models/yolo11m-pose32640.engine'):
        # Загружаем модели
        #self.model = YOLO(model_path, task='detect')
        self.pose_model = YOLO(pose_path)
        # Настройки папок
        self.save_dir = "evidence_folder"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.shm = shared_memory.SharedMemory(name="cv_frame_buffer")
        # Создаем numpy-обертку над памятью
        self.shared_array = np.ndarray((200, 640, 640, 3), dtype=np.uint8, buffer=self.shm.buf)
        # Словари для состояний
        self.r = redis.Redis(host='localhost', port=6379)
        self.last_save = {}
        self.phone_counters = {}
        self.cooldown = 3
        self.threshold = 3 # TEMPORAL_THRESHOLD
        self.detections = [] # Список для вывода в таблицу

        #stats
        self.frameCount = 0
        self.peopleAVG = 0
        self.peopleMax = 0

    def process_batch(self, frames, cam_ids):
        if not frames or any(f is None for f in frames):
            return frames, self.detections

        current_time = time.time()
        all_results = []
        all_pose_results = []
        display_time = time.strftime("%H:%M:%S")
        # 1. Пакетный инференс (RTX 5070 обработает frames параллельно)
        # Важно: imgsz=640 сильно ускорит процесс без потери качества для телефонов
        for i in range(0, len(frames), MAX_BATCH):
            micro_batch = frames[i : i + MAX_BATCH]

            # Инференс детекции
            # res = self.model.track(
            #     source=micro_batch,
            #     imgsz=640,
            #     half=True,
            #     conf=0.4,
            #     verbose=False,
            # )
            # all_results.extend(res) # Собираем результаты в один список

            # Инференс поз
            pose_results = self.pose_model.track(
                source=micro_batch,
                persist=True,
                conf=0.05,      # Позволь трекеру видеть "слабые" скелеты
                iou=0.5,       # Поможет не склеивать детей, сидящих рядом
                imgsz=640,
                half=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            all_pose_results.extend(pose_results)


        processed_output = []

        # 2. Итерируемся по кадрам и соответствующим результатам, отправляем пачкой через pipeline
        pipe = self.r.pipeline()

        for i, res in enumerate(all_pose_results):
            # 1. Выбираем индекс в кольцевом буфере (от 0 до 199)
            # 1. Выбираем индекс в кольцевом буфере (от 0 до 199)
            shm_idx = self.frameCount % 200
            # 2. Мгновенно копируем кадр в общую память
            # 2. Мгновенно копируем кадр в общую память
            self.shared_array[shm_idx][:] = frames[i][:]

            boxes = (
                res.boxes.data.cpu().numpy().astype(np.float16)
                if getattr(res, "boxes", None) is not None
                else np.empty((0, 6), dtype=np.float16)
            )
            kpts = (
                res.keypoints.data.cpu().numpy().astype(np.float16)
                if getattr(res, "keypoints", None) is not None
                else np.empty((0, 3, 17), dtype=np.float16)
            )

            meta = {
                "idx": shm_idx,
                "cid": cam_ids[i] if i < len(cam_ids) else "unknown",
                "box": boxes.tobytes(),
                "box_shape": boxes.shape,
                "kpt": kpts.tobytes(),
                "kpt_shape": kpts.shape,
            }

            binary_data = msgpack.packb(meta, use_bin_type=True)
            pipe.rpush("raw_ai_results", binary_data)

            self.frameCount += 1

        pipe.execute()

        return processed_output, self.detections
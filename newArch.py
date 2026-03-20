import json
import os

import numpy as np
import msgpack

gst_bin_path = r'D:\program\msvc_x86_64\bin'
import redis
# if os.path.exists(gst_bin_path):
#     os.add_dll_directory(gst_bin_path)
import cv2
import time
from ultralytics import YOLO
from multiprocessing import shared_memory
# model_det = YOLO('yolo11m.pt') # или твой путь к .pt
# model_det.export(format='engine', device=0, imgsz=960, half=True, batch=8, dynamic=True)
#
# Для модели поз (Pose) — ОБЯЗАТЕЛЬНО тоже в .engine!
# model_pose = YOLO('yolo11l-pose.pt')
# model_pose.export(format='engine', device=0, imgsz=640, half=True, batch=8, dynamic=True)
MAX_BATCH = 8

class ProctoringEngine:
    def __init__(self, pose_path='yolo11l-pose.engine'):
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
        # threshold теперь трактуем как "секунд подозрительности", а не "кол-во кадров"
        self.threshold = 2.0  # seconds
        self.detections = [] # Список для вывода в таблицу
        # Простая "анти-списывательная" логика на позе (по камере)
        self.pose_state = {}  # cam_id -> state dict

        #stats
        self.frameCount = 0
        self.peopleAVG = 0
        self.peopleMax = 0

    @staticmethod
    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _choose_main_person(boxes: np.ndarray, kpts: np.ndarray):
        """
        Выбираем "главного" человека на кадре (обычно сдающего) — по максимальной площади бокса.
        boxes: (N, 6) где [x1, y1, x2, y2, conf, cls]
        kpts: (N, K, 3)
        """
        if boxes is None or kpts is None or boxes.size == 0 or kpts.size == 0:
            return None
        n = min(boxes.shape[0], kpts.shape[0])
        if n <= 0:
            return None
        b = boxes[:n]
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        idx = int(np.argmax(areas))
        return idx

    @staticmethod
    def _pose_suspicion_from_kpts(person_kpts: np.ndarray):
        """
        Простейшие эвристики по COCO-ключевым точкам (17):
        0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder, ...
        Возвращает (is_suspicious: bool, reason: str)
        """
        if person_kpts is None or person_kpts.size == 0:
            return True, "no_pose"

        def kp(i):
            x, y, c = person_kpts[i]
            return float(x), float(y), float(c)

        nx, ny, nc = kp(0)
        lsx, lsy, lsc = kp(5)
        rsx, rsy, rsc = kp(6)

        # Если нет базовых точек — считаем это подозрительным (человек отвернулся/частично пропал)
        if nc < 0.3 or lsc < 0.3 or rsc < 0.3:
            return True, "low_face_or_shoulders"

        shoulder_w = abs(rsx - lsx)
        if shoulder_w < 1.0:
            return False, ""

        mid_x = (lsx + rsx) / 2.0
        mid_y = (lsy + rsy) / 2.0

        yaw_ratio = abs(nx - mid_x) / shoulder_w
        down_ratio = (ny - mid_y) / shoulder_w
        return yaw_ratio, down_ratio, shoulder_w

    def _update_pose_warning(self, cam_id: str, boxes: np.ndarray, kpts: np.ndarray, now_ts: float):
        st = self.pose_state.get(cam_id)
        if st is None:
            st = {
                "score_s": 0.0,          # накопленные "секунды подозрения"
                "last_warn_ts": 0.0,
                "last_ts": now_ts,
                "calib_end_ts": now_ts + 10.0,  # авто-калибровка 10 секунд
                "base_yaw": None,        # baseline yaw_ratio (EMA)
                "base_down": None,       # baseline down_ratio (EMA)
            }

        # dt для времени (защита от скачков/паузы)
        dt = max(0.0, min(0.25, float(now_ts - st.get("last_ts", now_ts))))
        st["last_ts"] = now_ts

        warn_text = ""
        suspicious = False
        reason = ""

        # Достаём главного человека и считаем признаки
        if kpts is None or kpts.size == 0:
            suspicious, reason = True, "no_pose"
            yaw_ratio = None
            down_ratio = None
        else:
            main_idx = self._choose_main_person(boxes, kpts)
            if main_idx is None:
                suspicious, reason = True, "no_person"
                yaw_ratio = None
                down_ratio = None
            else:
                try:
                    yaw_ratio, down_ratio, _ = self._pose_suspicion_from_kpts(kpts[main_idx])
                except Exception:
                    yaw_ratio = None
                    down_ratio = None
                    suspicious, reason = True, "bad_pose"

        # Авто-калибровка baseline: первые 10 секунд собираем EMA по "норме"
        if yaw_ratio is not None and down_ratio is not None and now_ts <= float(st["calib_end_ts"]):
            alpha = 0.10
            if st["base_yaw"] is None:
                st["base_yaw"] = float(yaw_ratio)
            else:
                st["base_yaw"] = (1 - alpha) * float(st["base_yaw"]) + alpha * float(yaw_ratio)

            if st["base_down"] is None:
                st["base_down"] = float(down_ratio)
            else:
                st["base_down"] = (1 - alpha) * float(st["base_down"]) + alpha * float(down_ratio)

        # Пороги относительно baseline (если baseline ещё не набрали — fallback)
        base_yaw = st["base_yaw"]
        base_down = st["base_down"]
        yaw_thr = (float(base_yaw) + 0.25) if base_yaw is not None else 0.35
        down_thr = (float(base_down) + 0.15) if base_down is not None else 0.20

        if not suspicious and yaw_ratio is not None and down_ratio is not None:
            side_look = float(yaw_ratio) > yaw_thr
            head_down = float(down_ratio) > down_thr

            if side_look and head_down:
                suspicious, reason = True, "side+down"
            elif side_look:
                suspicious, reason = True, "side"
            elif head_down:
                suspicious, reason = True, "down"

        # Накопление по времени (секунды), а не по кадрам:
        # - если подозрительно: +dt
        # - если норм: -dt*0.6 (медленнее "прощаем", чем набираем)
        if suspicious:
            st["score_s"] = min(10.0, float(st["score_s"]) + dt)
        else:
            st["score_s"] = max(0.0, float(st["score_s"]) - dt * 0.6)

        if float(st["score_s"]) >= float(self.threshold) and (now_ts - float(st["last_warn_ts"])) >= float(self.cooldown):
            warn_text = f"suspicious_pose:{reason}"
            st["last_warn_ts"] = now_ts
            try:
                self.r.rpush("proctor_warnings", json.dumps({"ts": now_ts, "cam_id": cam_id, "type": warn_text}))
            except Exception:
                pass

        self.pose_state[cam_id] = st
        return warn_text, float(st["score_s"])

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

            cam_id = cam_ids[i] if i < len(cam_ids) else "unknown"
            warn_text, pose_score = self._update_pose_warning(str(cam_id), boxes, kpts, current_time)

            meta = {
                "idx": shm_idx,
                "cid": cam_id,
                "box": boxes.tobytes(),
                "box_shape": boxes.shape,
                "kpt": kpts.tobytes(),
                "kpt_shape": kpts.shape,
                "warn": warn_text,
                "pose_score": pose_score,
            }

            binary_data = msgpack.packb(meta, use_bin_type=True)
            pipe.rpush("raw_ai_results", binary_data)

            self.frameCount += 1

        pipe.execute()

        return processed_output, self.detections
import cv2
import numpy as np
import time
import os
from multiprocessing import shared_memory
import redis
import json

class VisualizerWorker:
    def __init__(self, shm_name, frame_shape):
        # Подключаемся к общей памяти
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=self.shm.buf)

        self.r = redis.Redis(host='localhost', port=6379)
        self.save_dir = "evidence_folder"
        self.phone_counters = {}
        self.last_save = {}
        self.threshold = 3
        self.cooldown = 5

    def run(self):
        print("Визуализатор запущен...")
        while True:
            # Забираем только метаданные из Redis
            data = self.r.lpop("visualize_queue")
            if not data:
                time.sleep(0.01)
                continue

            task = json.loads(data)
            # Извлекаем данные (координаты уже посчитаны ИИ)
            frame_idx = task['idx'] # Позиция кадра в Shared Memory
            current_persons = task['persons']
            phones = task['phones']
            keypoints = task['keypoints']
            cam_id = task['cam_id']

            # Ссылка на конкретный кадр в памяти
            frame = self.frame_buffer[frame_idx].copy()

            # 1. Рисуем скелеты (Keypoints)
            for kpt_set in keypoints:
                for pt in kpt_set:
                    x, y, conf = pt
                    if conf > 0.3:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

            # 2. Логика телефонов
            for s_id, p_box in current_persons.items():
                phone_in_zone = False
                for ph in phones:
                    ph_c = [(ph[0]+ph[2])/2, (ph[1]+ph[3])/2]
                    roi_y2 = p_box[3] + int((p_box[3]-p_box[1])*0.4)

                    if (p_box[0]-40 < ph_c[0] < p_box[2]+40) and (p_box[1] < ph_c[1] < roi_y2):
                        phone_in_zone = True
                        cv2.rectangle(frame, (ph[0], ph[1]), (ph[2], ph[3]), (0,0,255), 2)

                # Обновляем счетчик
                cnt = self.phone_counters.get(s_id, 0)
                self.phone_counters[s_id] = cnt + 1 if phone_in_zone else max(0, cnt - 1)

                # Сохранение нарушения
                if self.phone_counters[s_id] >= self.threshold:
                    cur_t = time.time()
                    if cur_t - self.last_save.get(s_id, 0) > self.cooldown:
                        fname = f"cam{cam_id}_id{s_id}_{int(cur_t)}.jpg"
                        cv2.imwrite(os.path.join(self.save_dir, fname), frame)
                        self.last_save[s_id] = cur_t
                        print(f"!!! НАРУШЕНИЕ ЗАПИСАНО: {fname}")

                    cv2.putText(frame, "PHONE!", (int(p_box[0]), int(p_box[3]+20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # (Опционально) Показать превью
            cv2.imshow("Live Monitor", frame)
            cv2.waitKey(1)
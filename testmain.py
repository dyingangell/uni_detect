import cv2
import time
import os
from ultralytics import YOLO
# model_det = YOLO('yolo11m.pt') # или твой путь к .pt
# model_det.export(format='engine', device=0, imgsz=960, half=True, batch=8, dynamic=True)
#
## Для модели поз (Pose) — ОБЯЗАТЕЛЬНО тоже в .engine!
model_pose = YOLO('yolo11m-pose.pt')
model_pose.export(format='engine', device=0, imgsz=640, half=True, batch=16, dynamic=True)

class ProctoringEngine:
    def __init__(self, model_path='models/yolo11m4960.engine', pose_path='models/yolo11m-pose16640.engine'):
        # Загружаем модели
        #self.model = YOLO(model_path, task='detect')
        self.pose_model = YOLO(pose_path)

        # Настройки папок
        self.save_dir = "evidence_folder"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Словари для состояний
        self.last_save = {}
        self.phone_counters = {}
        self.cooldown = 3
        self.threshold = 3 # TEMPORAL_THRESHOLD
        self.detections = [] # Список для вывода в таблицу

        #stats
        self.frameCount = 0
        self.peopleAVG = 0
        self.peopleMax = 0
    def get_pose_status(self, kpts_data, box):
        pts = kpts_data.xy[0].cpu().numpy()
        conf = kpts_data.conf[0].cpu().numpy()
        if conf[0] < 0.25 or conf[5] < 0.25 or conf[6] < 0.25:
            return "Normal", (0, 255, 0)
        nose = pts[0]
        shoulder_mid = (pts[5] + pts[6]) / 2
        person_height = abs(box[3] - box[1])
        neck_ratio = (nose[1] - shoulder_mid[1]) / person_height
        return ("SUSPICIOUS", (0, 0, 255)) if neck_ratio > 0.08 else ("Normal", (0, 255, 0))

    def process_batch(self, frames):
        if not frames or any(f is None for f in frames):
            return frames
        current_time = time.time()
        MAX_BATCH = 16
        all_results = []
        all_pose_results = []
        display_time = time.strftime("%H:%M:%S")
        stream_ids = list(range(len(frames)))
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
                verbose=True,
            )
            all_pose_results.extend(pose_results)


        processed_output = []

        # 2. Итерируемся по кадрам и ИХ СООТВЕТСТВУЮЩИМ результатам
        for i, frame in enumerate(frames):
            # БЕРЕМ РЕЗУЛЬТАТ ИМЕННО ДЛЯ ТЕКУЩЕГО КАДРА [i]
            # res = all_results[i]
            pose_res = all_pose_results[i]

            current_persons = {}
            phones = []

            # if res.boxes.id is not None:
            #     ids = res.boxes.id.int().tolist()
            #     cls = res.boxes.cls.int().tolist()
            #     boxes = res.boxes.xyxy.int().tolist()
            #     for obj_id, obj_cls, box in zip(ids, cls, boxes):
            #         # Добавляем префикс камеры (например, 1001, 2001)
            #         unique_id = obj_id + (i + 1) * 1000
            #
            #         if obj_cls == 0:
            #             current_persons[unique_id] = box
            #         elif obj_cls == 67:
            #             phones.append(box)

            # 3. Обработка поз для конкретного кадра
            pose_statuses = {}
            if pose_res.keypoints is not None:
                self.frameCount += 1
                # Считаем, сколько людей именно В ЭТОМ кадре
                people_in_current_frame = len(pose_res.keypoints)

                # Обновляем среднее: (старое_среднее * кол-во_прошлых_кадров + люди_сейчас) / новый_счетчик
                self.peopleAVG = (self.peopleAVG * (self.frameCount - 1) + people_in_current_frame) / self.frameCount
                if self.peopleMax < self.peopleAVG:
                    self.peopleMax = self.peopleAVG
                # В YOLO Pose результаты для нескольких людей лежат в .keypoints
                for kpts in pose_res.keypoints:
                    # Проверяем уверенность детекции ключевых точек
                    if kpts.conf is None or kpts.conf[0][0] < 0.2: continue
                    for pt in kpts.data[0].cpu().numpy():
                                x, y, c = pt
                                if c > 0.3: cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
                    nose_p = kpts.xy[0][0].cpu().numpy()
            if self.frameCount % 100 == 0:
                print(f"Stats: AVG: {self.peopleAVG:.2f} | MAX - {self.peopleMax}| Frames: {self.frameCount}")
            # 4. Основная логика нарушений
            for s_id, p_box in current_persons.items():
                status, color = pose_statuses.get(s_id, ("Normal", (0, 255, 0)))
                # cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), color, 2)
                # cv2.putText(frame, f"ID {s_id} {status}", (p_box[0], p_box[1]-10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                phone_in_zone = False
                for ph in phones:
                    ph_c = [(ph[0]+ph[2])/2, (ph[1]+ph[3])/2]
                    roi_y2 = p_box[3] + int((p_box[3]-p_box[1])*0.4)
                    if (p_box[0]-40 < ph_c[0] < p_box[2]+40) and (p_box[1] < ph_c[1] < roi_y2):
                        phone_in_zone = True
                        cv2.rectangle(frame, (ph[0], ph[1]), (ph[2], ph[3]), (0,0,255), 2)
                        break

                cnt = self.phone_counters.get(s_id, 0)
                self.phone_counters[s_id] = cnt + 1 if phone_in_zone else max(0, cnt - 1)

                if self.phone_counters[s_id] >= self.threshold:
                    # Логика сохранения (кулдаун)
                    if current_time - self.last_save.get(s_id, 0) > self.cooldown:
                        file_name = f"cheat_id{s_id}_{time.strftime('%H%M%S')}.jpg"
                        cv2.imwrite(os.path.join(self.save_dir, file_name), frame)
                        self.last_save[s_id] = current_time
                        self.detections.append({"Time": display_time, "Student": s_id, "Status": "Phone"})

                    cv2.putText(frame, "DETECTED!", (p_box[0], p_box[3]+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            processed_output.append(frame)

        return processed_output, self.detections
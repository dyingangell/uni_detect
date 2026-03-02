import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from ultralytics.utils.plotting import Annotator



# 1. Загружаем модель
pose_model = YOLO('yolo11n-pose.pt')
model = YOLO('yolo11x.engine', task='detect')


video_path = "test3.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

#обрезка ненужных кадров
target_fps = 15

frame_skip = int(fps / target_fps)
if frame_skip < 1:
    frame_skip = 1
frame_count = 0

detections = []

# Создаем папку для нарушителей, если её нет
SAVE_DIR = "evidence_folder"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

last_save_per_student = {}
COOLDOWN_SECONDS = 1

phone_frames_counter = {}
TEMPORAL_THRESHOLD = 5


print("Начинаю анализ... Нажмите 'q' для выхода.")





def get_pose_status(keypoints_data, person_box):
    """
    Проверяет, наклонена ли голова студента слишком низко.
    keypoints_data: тензор ключевых точек от YOLO Pose
    """
    if keypoints_data is None or len(keypoints_data.xy[0]) < 7:
        return "Normal", (0, 255, 0)

    pts = keypoints_data.xy[0].cpu().numpy()
    nose = pts[0]        # Нос
    l_shoulder = pts[5]  # Левое плечо
    r_shoulder = pts[6]  # Правое плечо
    conf = keypoints_data.conf[0].cpu().numpy()
    if conf[0] < 0.5 or conf[5] < 0.5: # Если нос или плечо плохо видны
        return "Normal", (0, 255, 0)
    # Если точки не детектированы (нули), выходим
    if nose[1] == 0 or l_shoulder[1] == 0:
        return "Normal", (0, 255, 0)

    shoulder_y_avg = (l_shoulder[1] + r_shoulder[1]) / 2

    # ЛОГИКА: Если нос ниже линии плеч на 5 пикселей (наклон к телефону на коленях)
    if nose[1] > shoulder_y_avg + 5:
        return "SUSPICIOUS POSE", (0, 0, 255)

    return "Normal", (0, 255, 0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Используем time.time() для математики, а strftime — для имен файлов
    current_time_seconds = time.time()
    display_time = time.strftime("%H:%M:%S")


    # 1. Детекция и трекинг
    results = model.track(
    source=frame,
    imgsz=1280,
    persist=True,
    classes=[0, 67],
    conf=0.3,
    iou=0.4,
    tracker="botsort.yaml",
    half=True,
    augment=False,
    verbose=False
    )
    # 2. Детекция поз (Скелеты) - запускаем на всем кадре
    pose_results = pose_model(frame, imgsz=640, verbose=False, conf=0.3)

    current_persons = {}
    phones = []

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.int().tolist()
        cls = results[0].boxes.cls.int().tolist()
        boxes = results[0].boxes.xyxy.int().tolist()

        for obj_id, obj_cls, box in zip(ids, cls, boxes):
            if obj_cls == 0:
                current_persons[obj_id] = box
            elif obj_cls == 67:
                phones.append(box)

    # Анализ поз (сопоставляем скелеты с нашими ID студентов)
    pose_statuses = {}
    suspicious_kpts = []
    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:

        # Создаем аннотатор
        annotator = Annotator(frame, line_width=2)

        # Перебираем ВСЕ найденные скелеты БЕЗ привязки к ID для теста
        for kpts in pose_results[0].keypoints:
            # Получаем координаты носа в формате [x, y]
            nk = kpts.xy[0][0].cpu().numpy()

            # Если координаты не нулевые, рисуем скелет ВСЕГДА
            if nk[0] > 0 and nk[1] > 0:
                # Рисуем скелет стандартным методом
                annotator.kpts(kpts.data[0])

                # Привязка статуса к ID (оставляем для логики)
                for s_id, s_box in current_persons.items():
                    if s_box[0] < nk[0] < s_box[2] and s_box[1] < nk[1] < s_box[3]:
                        status, color = get_pose_status(kpts, s_box)
                        pose_statuses[s_id] = (status, color)

        # Перезаписываем кадр результатом аннотатора
        frame = annotator.result()
    for student_id, p_box in current_persons.items():
        px1, py1, px2, py2 = p_box
        p_status, p_color = pose_statuses.get(student_id, ("Normal", (0, 255, 0)))
        cv2.rectangle(frame, (px1, py1), (px2, py2), p_color, 2)
        cv2.putText(frame, f"ID {student_id} {p_status}", (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, p_color, 2)
        # Сначала проверяем кулдаун (30 сек)
        if student_id in last_save_per_student:
            if current_time_seconds - last_save_per_student[student_id] < COOLDOWN_SECONDS:
                continue

        roi_y2 = py2 + int((py2 - py1) * 0.4) # Расширяем зону вниз чуть больше

        # Флаг: видит ли этот конкретный студент телефон в ЭТОМ кадре
        phone_detected_this_frame = False
        detected_ph_coords = None

        for ph in phones:
            phx1, phy1, phx2, phy2 = ph
            cv2.rectangle(frame, (phx1, phy1), (phx2, phy2), (255, 0, 0), 2)

            cv2.putText(frame, "Phone", (phx1, phy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            ph_center_x = (ph[0] + ph[2]) / 2
            ph_center_y = (ph[1] + ph[3]) / 2
            if (px1 < ph_center_x < px2) and (py1 < ph_center_y < roi_y2):
                phone_detected_this_frame = True
                detected_ph_coords = ph
                break
        # Объединенный триггер инцидента
        incident_trigger = phone_detected_this_frame or (p_status == "SUSPICIOUS POSE")
        # --- ЛОГИКА СЧЕТЧИКА (вне цикла phones) ---
        if student_id not in phone_frames_counter:
            phone_frames_counter[student_id] = 0

        if phone_detected_this_frame:
            phone_frames_counter[student_id] += 1
        else:
            # Плавное затухание счетчика (чтобы не сбрасывать при мерцании кадра)
            phone_frames_counter[student_id] = max(0, phone_frames_counter[student_id] - 1)

        # ПРОВЕРКА ПОРОГА (например, 3 кадров подряд)
        if phone_frames_counter[student_id] >= TEMPORAL_THRESHOLD:
            # СБРАСЫВАЕМ СЧЕТЧИК, чтобы зафиксировать инцидент
            phone_frames_counter[student_id] = 0

            # Создаем кроп
            crop_y1, crop_y2 = max(0, py1 - 50), min(frame.shape[0], roi_y2 + 50)
            crop_x1, crop_x2 = max(0, px1 - 50), min(frame.shape[1], px2 + 50)
            evidence_crop = frame[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]

            # Сохранение файла
            file_ts = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"cheat_id{student_id}_{file_ts}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR, file_name), evidence_crop)

            # Обновляем кулдаун
            last_save_per_student[student_id] = current_time_seconds

            # Логируем
            detections.append({
                "video": video_path,
                "time": display_time,
                "student_id": student_id,
                "file": file_name
            })

            print(f"!!! СОХРАНЕНО: Студент {student_id} на {display_time}")

            # Отрисовка (для визуализации нарушения в реальном времени)
            if detected_ph_coords:
                cv2.rectangle(frame, (int(detected_ph_coords[0]), int(detected_ph_coords[1])),
                             (int(detected_ph_coords[2]), int(detected_ph_coords[3])), (0, 0, 255), 3)


    # 3. Финальный показ
    cv2.imshow("AI Proctoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 3. Сохраняем "базу" в CSV
df = pd.DataFrame(detections)
df.to_csv("cheating_report.csv", index=False)
print("Готово! Отчет сохранен в cheating_report.csv")
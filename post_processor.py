import os
from multiprocessing import shared_memory

gst_bin_path = r'D:\program\msvc_x86_64\bin'
if os.path.exists(gst_bin_path):
    os.add_dll_directory(gst_bin_path)

import cv2
import numpy as np
import redis
import msgpack
import time
import streamlit as st

# Подключаемся к той же памяти
shm = shared_memory.SharedMemory(name="cv_frame_buffer")
shared_array = np.ndarray((200, 640, 640, 3), dtype=np.uint8, buffer=shm.buf)
r = redis.Redis(host='localhost', port=6379)
r.delete("raw_ai_results")

num_cams = 64
cols = st.columns(4)
cam_placeholders = {}

print("Post-processor готов к работе...")
for cam_id in range(1, num_cams + 1):
    col = cols[(cam_id - 1) % 4]
    with col:
        st.markdown(f"**Cam {cam_id}**")
        cam_placeholders[str(cam_id)] = st.image(
            np.zeros((640, 640, 3), dtype=np.uint8), channels="BGR", width=320
        )

while True:
    data = r.lpop("raw_ai_results")
    if not data:
        time.sleep(0.01)
        continue

    meta = msgpack.unpackb(data, raw=False)

    idx = meta["idx"]
    cam_id = str(meta["cid"])
    warn_text = meta.get("warn", "") or ""
    pose_score = meta.get("pose_score", None)

    # Восстановление тензоров при необходимости
    boxes = np.frombuffer(meta["box"], dtype=np.float16).reshape(meta["box_shape"])
    kpts = np.frombuffer(meta["kpt"], dtype=np.float16).reshape(meta["kpt_shape"])

    frame = shared_array[idx].copy()
    if warn_text:
        score_str = f"{float(pose_score):.1f}" if pose_score is not None else "?"
        cv2.putText(
            frame,
            f"WARNING: {warn_text} (score={score_str})",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    # kpts.shape: (num_people, num_points, 3) -> (x, y, conf)
    if kpts.size > 0:
        num_people = kpts.shape[0]
        num_joints = kpts.shape[1]
        # Связи между точками (для линий) — пример для COCO-подобной разметки
        skeleton_pairs = [
            (5, 6),  # плечи
            (5, 7), (7, 9),  # левая рука
            (6, 8), (8, 10), # правая рука
            (11, 12),        # бёдра
            (11, 13), (13, 15),
            (12, 14), (14, 16),
        ]
        for p in range(num_people):
            person_kpts = kpts[p]  # shape: (num_joints, 3)
            # Точки
            for j in range(num_joints):
                x, y, conf = person_kpts[j]
                if conf < 0.3:
                    continue
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            # Линии между ключевыми точками
            for j1, j2 in skeleton_pairs:
                if j1 >= num_joints or j2 >= num_joints:
                    continue
                x1, y1, c1 = person_kpts[j1]
                x2, y2, c2 = person_kpts[j2]
                if c1 < 0.3 or c2 < 0.3:
                    continue
                cv2.line(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    2,
                )
    if cam_id in cam_placeholders:
        cam_placeholders[cam_id].image(frame, channels="BGR", width=320)
import os
import time
import threading
from collections import deque
from multiprocessing import shared_memory

gst_bin_path = r"D:\program\msvc_x86_64\bin"
if os.path.exists(gst_bin_path):
    os.add_dll_directory(gst_bin_path)

import cv2
import numpy as np
import redis
import msgpack


QUEUE_NAME = "raw_ai_results"
SHM_NAME = "cv_frame_buffer"
SHM_SHAPE = (200, 640, 640, 3)
REDIS_BATCH = 64
WARNINGS_KEY = "proctor_warnings"

# Связи между точками (для линий) — под COCO-подобную разметку (как в post_processor.py)
SKELETON_PAIRS = [
    (5, 6),  # плечи
    (5, 7),
    (7, 9),  # левая рука
    (6, 8),
    (8, 10),  # правая рука
    (11, 12),  # бёдра
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def _ensure_size(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if img.shape[1] == w and img.shape[0] == h:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _draw_pose(frame: np.ndarray, kpts: np.ndarray, conf_thr: float = 0.3) -> np.ndarray:
    """
    kpts.shape: (num_people, num_points, 3) -> (x, y, conf)
    Рисуем точки и линии как в streamlit post_processor.
    """
    if frame is None or kpts is None or kpts.size == 0:
        return frame

    num_people = kpts.shape[0]
    num_joints = kpts.shape[1]

    for p in range(num_people):
        person_kpts = kpts[p]
        # Точки
        for j in range(num_joints):
            x, y, conf = person_kpts[j]
            if conf < conf_thr:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Линии
        for j1, j2 in SKELETON_PAIRS:
            if j1 >= num_joints or j2 >= num_joints:
                continue
            x1, y1, c1 = person_kpts[j1]
            x2, y2, c2 = person_kpts[j2]
            if c1 < conf_thr or c2 < conf_thr:
                continue
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    return frame


def _make_mosaic(frames_by_cam: dict, cam_order: list, cols: int, tile_w: int, tile_h: int) -> np.ndarray:
    if not cam_order:
        return np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    rows = int(np.ceil(len(cam_order) / cols))
    mosaic = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for idx, cam_id in enumerate(cam_order):
        r = idx // cols
        c = idx % cols
        tile = frames_by_cam.get(cam_id)
        tile = _ensure_size(tile, tile_w, tile_h)
        y0, y1 = r * tile_h, (r + 1) * tile_h
        x0, x1 = c * tile_w, (c + 1) * tile_w
        mosaic[y0:y1, x0:x1] = tile

        cv2.putText(
            mosaic,
            f"Cam {cam_id}",
            (x0 + 8, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return mosaic


def _render_warnings_panel(warnings: deque, w: int = 520, h: int = 360) -> np.ndarray:
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        panel,
        "Warnings (latest first)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    y = 60
    for item in list(warnings)[:10]:
        cv2.putText(
            panel,
            item,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 26
        if y > h - 10:
            break
    return panel


def _run_cameras_loop(stop_evt: threading.Event):
    r = redis.Redis(host="localhost", port=6379)
    # Стартуем "с чистого листа", чтобы не показывать старые варнинги/кадры из очереди
    try:
        r.delete(QUEUE_NAME)
    except Exception:
        pass

    shm = shared_memory.SharedMemory(name=SHM_NAME)
    shared_array = np.ndarray(SHM_SHAPE, dtype=np.uint8, buffer=shm.buf)

    frames_by_cam = {}
    cam_order = []

    cv2.namedWindow("Cameras", cv2.WINDOW_NORMAL)

    # Размеры тайла меньше, чем 640, чтобы сетка влезала на экран
    # Пагинация: 4 камеры на страницу (2x2)
    cols = 2
    tile_w, tile_h = 320, 320
    last_render = 0.0
    selected_cam: str | None = None
    mosaic_rows = 1
    page = 0
    per_page = 4

    def on_mouse(event, x, y, _flags, _param):
        nonlocal selected_cam, page
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not cam_order:
            return
        # Если уже в "зуме" — клик возвращает назад
        if selected_cam is not None:
            selected_cam = None
            return
        # Вычисляем индекс плитки по координатам клика
        c = int(x // tile_w)
        r_ = int(y // tile_h)
        idx = r_ * cols + c
        start = page * per_page
        end = min(len(cam_order), start + per_page)
        page_cams = cam_order[start:end]
        if idx < 0 or idx >= len(page_cams):
            return
        selected_cam = page_cams[idx]

    cv2.setMouseCallback("Cameras", on_mouse)

    try:
        while not stop_evt.is_set():
            # Батч-забор из Redis (меньше накладных расходов, меньше лагов UI)
            pipe = r.pipeline()
            pipe.lrange(QUEUE_NAME, 0, REDIS_BATCH - 1)
            pipe.ltrim(QUEUE_NAME, REDIS_BATCH, -1)
            raw_batch, _ = pipe.execute()

            if not raw_batch:
                time.sleep(0.005)
            else:
                for data in raw_batch:
                    try:
                        meta = msgpack.unpackb(data, raw=False)
                    except Exception:
                        continue

                    idx = int(meta.get("idx", 0))
                    cam_id = str(meta.get("cid", "unknown"))
                    warn_text = meta.get("warn", "") or ""
                    pose_score = meta.get("pose_score", None)

                    frame = shared_array[idx].copy()

                    # Восстановим keypoints и нарисуем скелет
                    try:
                        kpt_shape = meta.get("kpt_shape", None)
                        kpt_bytes = meta.get("kpt", None)
                        if kpt_shape and kpt_bytes:
                            kpts = np.frombuffer(kpt_bytes, dtype=np.float16).reshape(kpt_shape)
                            frame = _draw_pose(frame, kpts, conf_thr=0.3)
                    except Exception:
                        pass

                    # Нарисуем warning на самом тайле, чтобы видно было даже в сетке
                    if warn_text:
                        score_str = f"{float(pose_score):.1f}" if pose_score is not None else "?"
                        cv2.putText(
                            frame,
                            f"WARNING: {warn_text} ({score_str})",
                            (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    frames_by_cam[cam_id] = frame
                    if cam_id not in cam_order:
                        cam_order.append(cam_id)

            # Рендерим ~30 FPS (и не зависим от частоты прихода меты)
            now = time.time()
            if now - last_render >= 1.0 / 30.0:
                total_pages = max(1, int(np.ceil(len(cam_order) / per_page))) if cam_order else 1
                if page < 0:
                    page = 0
                if page >= total_pages:
                    page = total_pages - 1

                start = page * per_page
                end = min(len(cam_order), start + per_page)
                page_cams = cam_order[start:end]

                mosaic_rows = int(np.ceil(len(page_cams) / cols)) if page_cams else 1
                mosaic = _make_mosaic(frames_by_cam, page_cams, cols=cols, tile_w=tile_w, tile_h=tile_h)

                # Подсказка по управлению
                hint = "Click tile: zoom/back. A/D or Left/Right: pages. Q/Esc: quit."
                cv2.putText(
                    mosaic,
                    hint,
                    (10, mosaic.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    mosaic,
                    f"Page {page + 1}/{total_pages} (cams {start + 1}-{end})",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if selected_cam is None:
                    cv2.imshow("Cameras", mosaic)
                else:
                    # Полноэкранный режим выбранной камеры
                    frame = frames_by_cam.get(selected_cam)
                    frame = _ensure_size(frame, 640, 640)
                    cv2.putText(
                        frame,
                        f"Cam {selected_cam} (zoom) - click to back",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Cameras", frame)

                last_render = now

            # На Windows стрелки/функциональные клавиши корректнее читать через waitKeyEx
            key_ex = cv2.waitKeyEx(1)
            key = key_ex & 0xFF

            if key == ord("q") or key == 27:
                break
            if key == ord("b"):
                selected_cam = None
            # страницы
            if key in (ord("a"), ord("A"), ord("j"), ord("J")):  # prev
                if selected_cam is None:
                    page = max(0, page - 1)
            if key in (ord("d"), ord("D"), ord("l"), ord("L")):  # next
                if selected_cam is None:
                    page = page + 1
            # Стрелки / PgUp / PgDn (waitKeyEx возвращает platform-specific codes)
            # Частые коды Windows: Left=2424832, Right=2555904, PgUp=2162688, PgDn=2228224
            if key_ex in (2424832, 81, 65361, 63234):  # left
                if selected_cam is None:
                    page = max(0, page - 1)
            if key_ex in (2555904, 83, 65363, 63235):  # right
                if selected_cam is None:
                    page = page + 1
            if key_ex in (2162688,):  # page up
                if selected_cam is None:
                    page = max(0, page - 1)
            if key_ex in (2228224,):  # page down
                if selected_cam is None:
                    page = page + 1
    finally:
        stop_evt.set()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            shm.close()
        except Exception:
            pass


def main():
    # Главное окно варнингов (скролл/фильтр/Show cheaters) — это Tkinter.
    # На Windows Tk лучше держать в главном потоке, а OpenCV-картинки крутить в фоне.
    import tkinter as tk
    from warnings_window import WarningsApp

    stop_evt = threading.Event()
    cam_thread = threading.Thread(target=_run_cameras_loop, args=(stop_evt,), daemon=True)
    cam_thread.start()

    root = tk.Tk()
    app = WarningsApp(root)

    # Перехват закрытия: закрываем и камеры тоже
    def on_close():
        stop_evt.set()
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    # Подождём корректного выхода камеры
    stop_evt.set()
    try:
        cam_thread.join(timeout=2.0)
    except Exception:
        pass


if __name__ == "__main__":
    main()


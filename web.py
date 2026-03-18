import threading
import time

import numpy as np
import streamlit as st
import cv2

# Настройка страницы
st.set_page_config(layout="wide")

frame_count = 0
SKIP_FRAMES = 1  # Обрабатываем каждый 3-й кадр (инференс 1 раз, затем 2 раза пропуск)
last_processed_frames = None

# 1. Загрузка движка
if 'engine' not in st.session_state:
    with st.spinner("Загружаем TensorRT..."):
        # Импортируем только здесь, чтобы ускорить запуск интерфейса
        from testmain import ProctoringEngine
        st.session_state.engine = ProctoringEngine()
        st.success("Движок инициализирован!")

class VideoStream:
    def __init__(self, src):
        self.src = src
        # Для чисел (камер) используем DSHOW, для файлов (строк) — авто
        if isinstance(self.src, int):
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.src)

        self.frame = None
        self.stopped = False

        # Сначала создаем поток, чтобы .start() всегда мог к нему обратиться
        self.t = threading.Thread(target=self.update, daemon=True)

        if not self.cap.isOpened():
            st.error(f"Не удалось открыть источник: {src}")
            self.stopped = True
            return

        # Настройка параметров (только для камер)
        if isinstance(self.src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30
        self.frame_delay = 1.0 / self.fps

    def start(self):
        # Запускаем только если источник открыт успешно
        if not self.stopped:
            self.t.start()
        return self

    def update(self):
        while not self.stopped:
            start_time = time.time()
            ret, frame = self.cap.read()

            if not ret:
                if isinstance(self.src, str): # Зацикливаем видео
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else: # Камера отвалилась
                    break

            self.frame = frame
            time_to_sleep = self.frame_delay - (time.time() - start_time)
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

# --- ИНТЕРФЕЙС ---
sources = ["video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4"]

# Используем checkbox вместо button для бесконечных процессов в Streamlit
run_watch = st.sidebar.checkbox("ЗАПУСТИТЬ МОНИТОРИНГ")

if run_watch:
    if 'streams' not in st.session_state:
        st.session_state.streams = [VideoStream(s).start() for s in sources]
        time.sleep(1)

    main_placeholder = st.empty()


    # Чтобы цикл не вешал интерфейс в 2026 году,
    # Streamlit рекомендует использовать небольшие паузы и явное обновление
    try:

        while run_watch:
            raw_frames = []
            for s in st.session_state.streams:
                f = s.read()
                if f is not None:
                    raw_frames.append(f)

            if len(raw_frames) > 15:
                # 1. ЛОГИКА ПРОПУСКА КАДРОВ
                if frame_count % SKIP_FRAMES == 0:
                    # Выполняем инференс
                    processed_frames, detections = st.session_state.engine.process_batch(raw_frames)
                    # Ресайзим сразу после обработки
                    resized_frames = [cv2.resize(f, (640, 360)) for f in processed_frames]

                    # 2. УНИВЕРСАЛЬНОЕ ПОСТРОЕНИЕ СЕТКИ (Grid)
                    cols = 4 if len(resized_frames) > 4 else 2
                    rows = int(np.ceil(len(resized_frames) / cols))

                    h, w, c = resized_frames[0].shape
                    black_screen = np.zeros((h, w, c), dtype=np.uint8)

                    # Заполняем пустые слоты, если камер, например, 7, а не 8
                    all_slots = list(resized_frames)
                    while len(all_slots) < rows * cols:
                        all_slots.append(black_screen)

                    # Склеиваем ряды
                    grid_rows = [np.hstack(all_slots[i*cols : (i+1)*cols]) for i in range(rows)]
                    last_display_grid = np.vstack(grid_rows)

                # 3. ВЫВОД (используем сохраненную сетку, если этот кадр пропускаем)
                if last_display_grid is not None:
                    # Исправляем ошибку из логов: используем width='stretch'
                    main_placeholder.image(last_display_grid, channels="BGR", width='stretch')

            frame_count += 1
            # Пауза для стабильности интерфейса
            time.sleep(0.01)
    finally:
        # Корректное завершение при снятии галочки
        if not run_watch and 'streams' in st.session_state:
            for s in st.session_state.streams:
                s.stop()
            del st.session_state.streams


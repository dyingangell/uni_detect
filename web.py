import threading
import time

import numpy as np
import streamlit as st
import cv2

# Page configuration
st.set_page_config(layout="wide")

frame_count = 0
SKIP_FRAMES = 2  # Process every 3rd frame (1 inference run, then skip 2 frames)
last_processed_frames = None

# 1. Engine initialization
if 'engine' not in st.session_state:
    with st.spinner("Loading TensorRT..."):
        # Import lazily here to speed up initial UI load
        from testmain import ProctoringEngine
        st.session_state.engine = ProctoringEngine()
        st.success("Engine initialized!")

class VideoStream:
    def __init__(self, src):
        self.src = src
        # Use DSHOW for numeric camera sources, default backend for file paths
        if isinstance(self.src, int):
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.src)

        self.frame = None
        self.stopped = False

        # Create the thread first so .start() can always reference it
        self.t = threading.Thread(target=self.update, daemon=True)

        if not self.cap.isOpened():
            st.error(f"Failed to open source: {src}")
            self.stopped = True
            return

        # Camera-only capture settings
        if isinstance(self.src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30
        self.frame_delay = 1.0 / self.fps

    def start(self):
        # Start only when the source was opened successfully
        if not self.stopped:
            self.t.start()
        return self

    def update(self):
        while not self.stopped:
            start_time = time.time()
            ret, frame = self.cap.read()

            if not ret:
                if isinstance(self.src, str): # Loop video input
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else: # Camera source dropped
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

# --- UI ---
sources = ["video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4","video/test7.mp4"]

# Use a checkbox instead of a button for long-running Streamlit loops
run_watch = st.sidebar.checkbox("START MONITORING")

if run_watch:
    if 'streams' not in st.session_state:
        st.session_state.streams = [VideoStream(s).start() for s in sources]
        time.sleep(1)

    main_placeholder = st.empty()


    # Keep the UI responsive: short sleeps plus explicit redraws
    try:

        while run_watch:
            raw_frames = []
            for s in st.session_state.streams:
                f = s.read()
                if f is not None:
                    raw_frames.append(f)

            if len(raw_frames) > 15:
                # 1. Frame skipping logic
                if frame_count % SKIP_FRAMES == 0:
                    # Run inference
                    processed_frames, detections = st.session_state.engine.process_batch(raw_frames)
                    # Resize right after processing
                    resized_frames = [cv2.resize(f, (640, 360)) for f in processed_frames]

                    # 2. Generic grid layout
                    cols = 4 if len(resized_frames) > 4 else 2
                    rows = int(np.ceil(len(resized_frames) / cols))

                    h, w, c = resized_frames[0].shape
                    black_screen = np.zeros((h, w, c), dtype=np.uint8)

                    # Fill empty slots (e.g., 7 cameras instead of 8)
                    all_slots = list(resized_frames)
                    while len(all_slots) < rows * cols:
                        all_slots.append(black_screen)

                    # Stack rows into a final grid
                    grid_rows = [np.hstack(all_slots[i*cols : (i+1)*cols]) for i in range(rows)]
                    last_display_grid = np.vstack(grid_rows)

                # 3. Render output (reuse cached grid on skipped frames)
                if last_display_grid is not None:
                    # Avoid the logged UI issue by using width='stretch'
                    main_placeholder.image(last_display_grid, channels="BGR", width='stretch')

            frame_count += 1
            # Small pause for UI stability
            time.sleep(0.01)
    finally:
        # Clean shutdown when the checkbox is turned off
        if not run_watch and 'streams' in st.session_state:
            for s in st.session_state.streams:
                s.stop()
            del st.session_state.streams


import streamlit as st
import redis
import numpy as np
import cv2
import time

st.set_page_config(layout="wide", page_title="AI Proctoring Monitor")
st.title("Monitoring System (16 Cameras)")

r = redis.Redis(host='localhost', port=6379)
camera_ids = [str(i) for i in range(1, 33)]

# Create an 8x4 grid
cols = st.columns(8)
placeholders = []
for i in range(32):
    with cols[i % 8]:
        st.write(f"Camera {camera_ids[i]}")
        placeholders.append(st.empty())

while True:
    for i, cam_id in enumerate(camera_ids):
        # Fetch processed frame from Redis
        img_bytes = r.get(f"result:{cam_id}")
        if img_bytes:
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholders[i].image(frame_rgb, use_container_width=True)

    time.sleep(0.01) # Small pause to keep the browser responsive

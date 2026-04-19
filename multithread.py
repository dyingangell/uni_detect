import os
import sys
import time
import threading
import json
import base64

gst_bin_path = r'D:\program\msvc_x86_64\bin'

if os.path.exists(gst_bin_path):
    os.add_dll_directory(gst_bin_path)

import cv2
import redis


def camera_worker(cam_id, video_file):
    # Skip GStreamer DLL checks for this test; use standard OpenCV backend
    r = redis.Redis(host='localhost', port=6379)

    if not os.path.exists(video_file):
        print(f"[Error] File not found: {video_file}")
        return

    # Open without GStreamer (plain file path)
    # This is typically much more stable for 32 file-based streams
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"[Error] OpenCV could not open file for Cam {cam_id}")
        return

    print(f"[Cam {cam_id}] Started successfully (Standard Backend)")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Resize to 640x640 (since GStreamer videoscale is not used)
        frame_resized = cv2.resize(frame, (1280 ,720))

        # Encode as JPEG
        _, img_encoded = cv2.imencode('.jpg', frame_resized)

        payload = {
            "cam_id": cam_id,
            "img": base64.b64encode(img_encoded.tobytes()).decode("ascii"),
        }
        r.lpush("image_batch_queue", json.dumps(payload))
        r.ltrim("image_batch_queue", 0, 1000)

        # Basic FPS throttling
        time.sleep(0.2)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python multithread.py <folder_num>")
        print("Example: python multithread.py 1")
        print("folder_num should be 1..8 (expects videos in video<folder_num>/test1.mp4 .. test8.mp4)")
        raise SystemExit(2)

    try:
        folder_num = int(sys.argv[1])  # For example: 1, 2, 3, or 4
    except ValueError:
        print(f"[Error] folder_num must be a number, got: {sys.argv[1]!r}")
        raise SystemExit(2)

    if folder_num < 1 or folder_num > 8:
        print(f"[Error] folder_num out of range 1..8, got: {folder_num}")
        raise SystemExit(2)

    threads = []
    for i in range(1, 9):
        # Compute unique camera ID for Redis
        # If folder_num = 1: 1..8
        # If folder_num = 2: 9..16
        # If folder_num = 3: 17..24
        # If folder_num = 4: 25..32
        unique_id = (folder_num - 1) * 8 + i

        # File path (keep i if files are named test1..test8 inside each folder)
        video_file = os.path.join(f"video{folder_num}", f"test{i}.mp4")

        # Pass unique_id as cam_id
        t = threading.Thread(target=camera_worker, args=(str(unique_id), video_file))
        t.daemon = True
        t.start()
        threads.append(t)
        time.sleep(0.5)

    print(f"Started {len(threads)} cameras. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")

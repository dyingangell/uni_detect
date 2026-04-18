import redis
import sys
import os

# Add path to GStreamer DLLs (this setup worked in local tests)
gst_bin_path = r'D:\program\msvc_x86_64\bin'
if os.path.exists(gst_bin_path):
    os.add_dll_directory(gst_bin_path)
import cv2

def start_producer(cam_id, source):
    r = redis.Redis(host='localhost', port=6379)

    # 1. Select source pipeline automatically
    if source.startswith("rtsp://"):
        # RTSP pipeline (with low-latency settings)
        gst_source = f'rtspsrc location={source} latency=0 protocols=tcp ! rtph264depay ! h264parse ! decodebin'
    else:
        # Pipeline for local files (e.g., test7.mp4)
        # filesrc reads the file; decodebin auto-detects the format
        gst_source = f'filesrc location="{source}" ! decodebin'

    # 2. Build final pipeline (shared processing stage)
    gst_pipeline = (
        f'{gst_source} ! '
        f'videoscale ! '
        f'video/x-raw, width=640, height=640 ! '
        f'videoconvert ! '
        f'appsink drop=True'
    )

    print(f"Starting pipeline: {gst_pipeline}")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print(f"Error: Failed to start GStreamer for camera {cam_id}")
        return

    print(f"Producer started for camera {cam_id} via GStreamer...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Connection lost for camera {cam_id}. Reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            continue

        # Push frame to Redis
        # Frame is already 640x640 from GStreamer, so no Python resize() is needed
        _, img_encoded = cv2.imencode('.jpg', frame)
        r.lpush("image_batch_queue", img_encoded.tobytes()) # Must push encoded JPEG bytes
        r.ltrim("image_batch_queue", 0, 100)
        print(f"[{cam_id}] Frame sent to Redis") # Keep for delivery/debug checks

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python producer.py <id> <rtsp_url>")
    else:
        start_producer(sys.argv[1], sys.argv[2])
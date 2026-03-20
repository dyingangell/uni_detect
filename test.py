import redis
import os
gst_bin_path = r'D:\program\msvc_x86_64\bin'

if os.path.exists(gst_bin_path):
    os.add_dll_directory(gst_bin_path)
import cv2
import numpy as np

r = redis.Redis(host='localhost', port=6379)

while True:
    # Читаем кадр 5-й камеры из Redis
    img_bytes = r.get("camera:5")
    if img_bytes:
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        cv2.imshow("Check Camera 5", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
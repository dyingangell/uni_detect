import subprocess
import sys
import time

# Test settings
NUM_CAMERAS = 32  # Start with 20 and scale up to 100
VIDEO_PATH = "video/test7.mp4" # Path to your test video file
PRODUCER_SCRIPT = "producer.py"
PYTHON_EXE = sys.executable
processes = []

print(f"Starting {NUM_CAMERAS} camera emulators...")

for i in range(1, NUM_CAMERAS + 1):
    # Start each producer as a separate background process
    # Command: python producer.py [ID] [PATH]
    time.sleep(0.1)
    p = subprocess.Popen([PYTHON_EXE, PRODUCER_SCRIPT, str(i), VIDEO_PATH])
    processes.append(p)



print(f"\nAll {NUM_CAMERAS} cameras are running.")
print("Press Ctrl+C to stop all tests.")

try:
    # Keep script alive while child processes are running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping all cameras...")
    for p in processes:
        p.terminate()
    print("Test finished.")

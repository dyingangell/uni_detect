import cv2
import pandas as pd
import os
from ultralytics import YOLO

# 1. Load YOLO for pose detection
model = YOLO('yolov8n-pose.pt')

# Container for all extracted rows
data = []


# Process all videos in a folder
def process_videos(folder_path, label):
    for filename in os.listdir(folder_path):
        if not filename.endswith(('.mp4', '.avi')): continue

        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break  # Video end

            # Run frame through YOLO
            results = model(frame, verbose=False)

            for r in results:
                # If a person and their skeleton are detected in the frame
                if r.keypoints is not None and len(r.keypoints) > 0:
                    # Use normalized coordinates (0..1) so subject scale
                    # does not destabilize the model
                    points = r.keypoints.xyn.cpu().numpy()

                    # YOLO returns 17 (x, y) points; flatten into one vector (34 values)
                    flat_points = points.flatten()

                    # Keep only complete 34-value samples to avoid corrupted frames
                    if len(flat_points) == 34:
                        row = list(flat_points)
                        row.append(label)  # Append class label: 0 or 1
                        data.append(row)

        cap.release()
        print(f"Video {filename} processed.")


print("Starting extraction of normal poses...")
process_videos('videos/normal', label=0)  # 0 = no cheating

print("Starting extraction of cheating poses...")
process_videos('videos/cheating', label=1)  # 1 = cheating

# Save dataset to a table (DataFrame)
columns = [f'point_{i}' for i in range(34)] + ['label']
df = pd.DataFrame(data, columns=columns)
df.to_csv('pose_dataset.csv', index=False)

print(f"Done! Collected {len(df)} frames. Saved to pose_dataset.csv")

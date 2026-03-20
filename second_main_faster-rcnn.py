from ultralytics import YOLO

model_pose = YOLO('yolo11m-pose.pt')
model_pose.export(format='engine', device=0, imgsz=640, half=True, batch=16, dynamic=True)
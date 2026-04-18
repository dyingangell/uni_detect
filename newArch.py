import json
import os

import numpy as np
import msgpack

gst_bin_path = r'D:\program\msvc_x86_64\bin'
import redis
# if os.path.exists(gst_bin_path):
#     os.add_dll_directory(gst_bin_path)
import cv2
import time
from ultralytics import YOLO
from multiprocessing import shared_memory
# model_det = YOLO('yolo11m.pt') # or your custom .pt path
# model_det.export(format='engine', device=0, imgsz=960, half=True, batch=8, dynamic=True)
#
# For the pose model, export to .engine as well (required).
#model_pose = YOLO('yolo11l-pose.pt')
#model_pose.export(format='engine', device=0, imgsz=1280, half=True, batch=1, dynamic=True)
MAX_BATCH = 1

class ProctoringEngine:
    def __init__(self, pose_path='yolo11l-pose.engine'):
        # Load models
        #self.model = YOLO(model_path, task='detect')
        self.pose_model = YOLO(pose_path)
        # Configure output directories
        self.save_dir = "evidence_folder"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.shm = shared_memory.SharedMemory(name="cv_frame_buffer")
        # Create a NumPy view over shared memory
        self.shared_array = np.ndarray((200, 640, 640, 3), dtype=np.uint8, buffer=self.shm.buf)
        # State containers
        self.r = redis.Redis(host='localhost', port=6379)
        self.last_save = {}
        self.phone_counters = {}
        self.cooldown = 3
        # Interpret threshold as "seconds of suspicious behavior", not frame count
        self.threshold = 5.0  # Seconds of suspicious behavior required before raising a warning
        self.detections = [] # Used for table/UI output
        # Pose-based anti-cheating state (per camera)
        self.pose_state = {}  # cam_id_track_id -> state dict
        self.warned_persons = set()  # Global set of already-warned identities (cam_id_track_id)
        self.start_time = time.time()  # Engine start time for video timeline calculations

        # Clear Redis queues on startup to avoid stale warnings/results
        try:
            self.r.delete("proctor_warnings")
            self.r.delete("raw_ai_results")
            print("[INFO] Redis queues cleared at startup")
        except Exception as e:
            print(f"[WARN] Failed to clear Redis queues: {e}")

        #stats
        self.frameCount = 0
        self.peopleAVG = 0
        self.peopleMax = 0

    @staticmethod
    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _pose_suspicion_from_kpts(person_kpts: np.ndarray):
        """
        Extract the nose position relative to the shoulder midpoint.
        COCO keypoints (17):
        0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear, 5 l_shoulder, 6 r_shoulder, ...

        Returns: (rel_nose_x, rel_nose_y, shoulder_w, confidence_ok)
        - rel_nose_x, rel_nose_y: nose position relative to shoulder midpoint (normalized by shoulder width)
        - shoulder_w: shoulder width in pixels
        - confidence_ok: True when all required keypoints are visible
        """
        if person_kpts is None or person_kpts.size == 0:
            return None, None, None, False

        def kp(i):
            x, y, c = person_kpts[i]
            return float(x), float(y), float(c)

        nx, ny, nc = kp(0)  # nx - x nose, ny - y nose, nc - confidence nose
        lsx, lsy, lsc = kp(5)  # lsx - x left shoulder, lsy - y left shoulder, lsc - confidence left shoulder
        rsx, rsy, rsc = kp(6)  # rsx - x right shoulder, rsy - y right shoulder, rsc - confidence right shoulder

        # If core keypoints are missing, we cannot compute a reliable position
        if nc < 0.3 or lsc < 0.3 or rsc < 0.3:
            return None, None, None, False

        shoulder_w = abs(rsx - lsx)
        if shoulder_w < 1.0:
            return None, None, None, False

        # Midpoint between shoulders
        mid_x = (lsx + rsx) / 2.0   # lsx -  left shoulder x, rsx -  right shoulder x
        mid_y = (lsy + rsy) / 2.0  # lsy -  left shoulder y, rsy  - right shoulder y

        # Relative nose position (normalized by shoulder width)
        rel_nose_x = (nx - mid_x) / shoulder_w
        rel_nose_y = (ny - mid_y) / shoulder_w

        return rel_nose_x, rel_nose_y, shoulder_w, True

    def _update_pose_warning_for_person(self, person_key: str, cam_id: str, track_id: int,
                                          person_kpts: np.ndarray, person_box: np.ndarray, now_ts: float):
        """
        Process one tracked person by track_id.
        person_key: unique key in format "cam_id_track_id"
        person_box: [x1, y1, x2, y2, conf, cls]
        """
        # Fast path: this identity has already been warned, so skip further checks
        if person_key in self.warned_persons:
            return ""

        st = self.pose_state.get(person_key)
        if st is None:
            st = {
                "score_s": 0.0,           # Accumulated "suspicion seconds"
                "last_warn_ts": 0.0,
                "last_ts": now_ts,
                "calib_end_ts": now_ts + 10.0,  # Auto-calibration window (10 seconds)
                "base_nose_x": None,      # Baseline nose X position (circle center)
                "base_nose_y": None,      # Baseline nose Y position (circle center)
                "base_radius": 0.35,      # Radius of the "normal zone" (smaller = more sensitive)
                "max_away_dist": 1.2,     # Threshold for "person moved far away" (do not warn)
                "is_away": False,         # Flag indicating the person is far away
                "track_id": track_id,
                # Walking detection fields
                "last_box_cx": None,      # Previous bbox center X
                "last_box_cy": None,      # Previous bbox center Y
                "walk_speed_threshold": 30.0,  # Pixels per frame threshold for walking
            }

        # Clamp dt to protect against timestamp jumps and long pauses
        dt = max(0.0, min(0.25, float(now_ts - st.get("last_ts", now_ts))))
        st["last_ts"] = now_ts

        warn_text = ""

        suspicious = False
        is_walking = False

        # Walking detection: estimate movement speed from bbox center displacement
        if person_box is not None and len(person_box) >= 4:
            box_cx = (float(person_box[0]) + float(person_box[2])) / 2.0
            box_cy = (float(person_box[1]) + float(person_box[3])) / 2.0

            if st["last_box_cx"] is not None and st["last_box_cy"] is not None:
                # Distance moved by bbox center
                move_dist = ((box_cx - st["last_box_cx"])**2 + (box_cy - st["last_box_cy"])**2) ** 0.5

                # Mark as walking when the bbox shifts significantly
                if move_dist > st["walk_speed_threshold"]:
                    is_walking = True

            # Store current bbox center
            st["last_box_cx"] = box_cx
            st["last_box_cy"] = box_cy

        # Compute nose position for this person
        rel_nose_x, rel_nose_y = None, None
        try:
            rel_nose_x, rel_nose_y, shoulder_w, conf_ok = self._pose_suspicion_from_kpts(person_kpts)
            if not conf_ok:
                # Not suspicious: skip this frame when keypoint confidence is insufficient
                rel_nose_x, rel_nose_y = None, None
        except Exception:
            # Not suspicious: skip this frame on pose parsing errors
            pass

        # Auto-calibrate baseline: collect EMA of nose center during first 10 seconds
        if rel_nose_x is not None and rel_nose_y is not None and now_ts <= float(st["calib_end_ts"]):
            alpha = 0.10
            if st["base_nose_x"] is None:
                st["base_nose_x"] = float(rel_nose_x)
                st["base_nose_y"] = float(rel_nose_y)
            else:
                st["base_nose_x"] = (1 - alpha) * float(st["base_nose_x"]) + alpha * float(rel_nose_x)
                st["base_nose_y"] = (1 - alpha) * float(st["base_nose_y"]) + alpha * float(rel_nose_y)

        # Check whether nose position exits the baseline circle (only after calibration)
        in_calibration = now_ts <= float(st["calib_end_ts"])

        if rel_nose_x is not None and rel_nose_y is not None:
            if st["base_nose_x"] is not None and st["base_nose_y"] is not None:
                # Distance from baseline center to current nose position
                dist = ((rel_nose_x - st["base_nose_x"])**2 + (rel_nose_y - st["base_nose_y"])**2) ** 0.5

                # If person moved very far away, suppress warnings and wait
                if dist > st["max_away_dist"]:
                    st["is_away"] = True  # Person moved away from the seat area
                    suspicious = False
                # If person returns after being away, rebuild baseline and re-calibrate
                elif st["is_away"] and dist <= st["base_radius"]:
                    st["base_nose_x"] = float(rel_nose_x)
                    st["base_nose_y"] = float(rel_nose_y)
                    st["calib_end_ts"] = now_ts + 10.0
                    st["is_away"] = False
                    st["score_s"] = 0.0
                    suspicious = False
                # Regular out-of-circle behavior triggers the only warning type (dist_warning)
                # Calibration must be completed first
                elif dist > st["base_radius"] and not st["is_away"] and not in_calibration:
                    suspicious = True

                # DEBUG: print state every 30 frames
                if self.frameCount % 30 == 0:
                    status = "CALIB" if in_calibration else ("SUSP!" if suspicious else "ok")
                    print(f"[POSE] {person_key}: dist={dist:.2f} radius={st['base_radius']:.2f} [{status}]")

        # Time-based accumulation
        if is_walking:
            # Person is walking: decay score faster
            st["score_s"] = max(0.0, float(st["score_s"]) - dt * 2.0)
        elif suspicious:
            # 1:1 accumulation (dt seconds -> dt score)
            st["score_s"] = min(15.0, float(st["score_s"]) + dt * 1.0)  # st["score_s"] = min(10.0, float(st["score_s"]) + dt * 2.0)
        else:
            # Decay score when behavior returns to normal
            st["score_s"] = max(0.0, float(st["score_s"]) - dt * 1.0)

        # DEBUG: print score roughly once per second
        if suspicious and int(st["score_s"]) != int(st.get("_last_print_score", -1)):
            st["_last_print_score"] = int(st["score_s"])
            print(f"[DEBUG] {person_key}: score={st['score_s']:.1f}/{self.threshold} (suspicious={suspicious})")

        # Emit warning only once per identity (guard already checked at function entry)
        if float(st["score_s"]) >= float(self.threshold):
            warn_text = "dist_warning"
            self.warned_persons.add(person_key)  # Add to global set to prevent repeat warnings

            # Compute video-relative and wall-clock timestamps
            video_time_sec = now_ts - self.start_time
            video_time_str = time.strftime("%H:%M:%S", time.gmtime(video_time_sec))
            current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

            # Print warning details to stdout
            print(f"\n{'='*60}")
            print(f"⚠️  CHEATING DETECTED!")
            print(f"{'='*60}")
            print(f"🎥 Camera:      {cam_id}")
            print(f"👤 Person ID:   {track_id}")
            print(f"⏱️  Video time:  {video_time_str}")
            print(f"📅 Real time:   {current_time_str}")
            print(f"📊 Type:        dist_warning")
            print(f"{'='*60}\n")

            warning_data = {
                "ts": now_ts,
                "cam_id": cam_id,
                "track_id": track_id,
                "type": "dist_warning",
                "video_time": video_time_str,
                "real_time": current_time_str
            }
            print(f"[DEBUG] Sending to Redis: {warning_data}")
            try:
                self.r.rpush("proctor_warnings", json.dumps(warning_data))
            except Exception as e:
                print(f"[ERROR] Redis error: {e}")

        self.pose_state[person_key] = st
        return warn_text

    def _update_pose_warning(self, cam_id: str, boxes: np.ndarray, kpts: np.ndarray,
                              track_ids: np.ndarray, now_ts: float):
        """
        Process all tracked people in a frame, each with an independent baseline circle.
        Returns: warnings_list (dist_warning only)
        """
        warnings = []

        if kpts is None or kpts.size == 0 or track_ids is None or len(track_ids) == 0:
            return warnings

        n_persons = min(len(kpts), len(track_ids), len(boxes) if boxes is not None else 0)

        for idx in range(n_persons):
            track_id = int(track_ids[idx])
            person_key = f"{cam_id}_{track_id}"
            person_kpts = kpts[idx]
            person_box = boxes[idx] if boxes is not None and idx < len(boxes) else None

            warn_text = self._update_pose_warning_for_person(
                person_key, cam_id, track_id, person_kpts, person_box, now_ts
            )

            # Only append dist_warning events
            if warn_text == "dist_warning":
                warnings.append({"track_id": track_id, "type": "dist_warning"})
                print(f"[ENGINE] ✅ Warning added to list: track_id={track_id}")

        return warnings

    def process_batch(self, frames, cam_ids):
        if not frames or any(f is None for f in frames):
            return frames, self.detections

        current_time = time.time()
        all_results = []
        all_pose_results = []
        display_time = time.strftime("%H:%M:%S")
        # 1. Batch inference (GPU processes frames in parallel)
        # Note: imgsz=640 significantly improves throughput with minimal quality trade-off
        for i in range(0, len(frames), MAX_BATCH):
            micro_batch = frames[i : i + MAX_BATCH]

            # Detection inference
            # res = self.model.track(
            #     source=micro_batch,
            #     imgsz=640,
            #     half=True,
            #     conf=0.4,
            #     verbose=False,
            # )
            # all_results.extend(res) # Aggregate results into a single list

            # Pose inference
            pose_results = self.pose_model.track(
                source=micro_batch,
                persist=True,
                conf=0.05,      # Keep low-confidence skeletons for tracker stability
                iou=0.5,       # Helps avoid merging nearby seated students
                imgsz=640,
                half=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            all_pose_results.extend(pose_results)


        processed_output = []

        # 2. Iterate through frames/results and push data via a Redis pipeline
        pipe = self.r.pipeline()

        for i, res in enumerate(all_pose_results):
            # 1. Pick index in ring buffer (0..199)
            # 1. Pick index in ring buffer (0..199)
            shm_idx = self.frameCount % 200
            # 2. Copy frame into shared memory immediately
            # 2. Copy frame into shared memory immediately
            self.shared_array[shm_idx][:] = frames[i][:]

            boxes = (
                res.boxes.data.cpu().numpy().astype(np.float16)
                if getattr(res, "boxes", None) is not None
                else np.empty((0, 6), dtype=np.float16)
            )
            kpts = (
                res.keypoints.data.cpu().numpy().astype(np.float16)
                if getattr(res, "keypoints", None) is not None
                else np.empty((0, 3, 17), dtype=np.float16)
            )

            # Extract track IDs from tracker results
            track_ids = (
                res.boxes.id.cpu().numpy().astype(np.int32)
                if getattr(res.boxes, "id", None) is not None
                else np.array([], dtype=np.int32)
            )

            cam_id = cam_ids[i] if i < len(cam_ids) else "unknown"
            warnings = self._update_pose_warning(str(cam_id), boxes, kpts, track_ids, current_time)

            meta = {
                "idx": shm_idx,
                "cid": cam_id,
                "box": boxes.tobytes(),
                "box_shape": boxes.shape,
                "kpt": kpts.tobytes(),
                "kpt_shape": kpts.shape,
                "track_ids": track_ids.tobytes(),
                "warnings": warnings,  # dist_warning only
            }

            binary_data = msgpack.packb(meta, use_bin_type=True)
            pipe.rpush("raw_ai_results", binary_data)

            self.frameCount += 1

        pipe.execute()

        return processed_output, self.detections
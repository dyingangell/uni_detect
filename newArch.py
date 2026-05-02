import json
import os

import numpy as np
import msgpack
import csv

gst_bin_path = r'D:\program\msvc_x86_64\bin'
import redis
# if os.path.exists(gst_bin_path):
#     os.add_dll_directory(gst_bin_path)
import cv2
import time
import math
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
        # Create a NumPy view over shared memory (H, W, C)
        self.shared_array = np.ndarray((200, 720, 1280, 3), dtype=np.uint8, buffer=self.shm.buf)
        # State containers
        self.r = redis.Redis(host='localhost', port=6379)
        self.last_save = {}
        self.phone_counters = {}
        self.cooldown = 3

        # ---- Pose anti-cheat tuning ("circle" / sensitivity) ----
        # You can override these without editing code via env vars:
        #   POSE_BASE_RADIUS, POSE_WARN_THRESHOLD_S, POSE_CALIB_S, POSE_MAX_AWAY_DIST, POSE_WALK_PX
        # Circle radius in normalized units (nose offset / shoulder width). Bigger => less sensitive.
        # tuned default: slightly larger radius to reduce small-tilt false positives
        self.pose_base_radius = float(os.getenv("POSE_BASE_RADIUS", "0.38"))
        # Seconds of suspicious behavior required before raising a warning. Bigger => fewer warnings.
        # tuned default: require longer sustained deviation before warning
        self.threshold = float(os.getenv("POSE_WARN_THRESHOLD_S", "10.0"))
        # Calibration window (seconds) when baseline is learned.
        self.pose_calib_s = float(os.getenv("POSE_CALIB_S", "10.0"))
        # If person is *too far* from baseline, treat as "away" and do not warn.
        self.pose_max_away_dist = float(os.getenv("POSE_MAX_AWAY_DIST", "1.2"))
        # Walking detection threshold (pixels per frame for bbox center).
        self.pose_walk_speed_threshold = float(os.getenv("POSE_WALK_PX", "30.0"))
        # Multiplier to scale combined excess into score accumulation (bigger => faster alerts)
        # tuned default: moderate accumulation speed
        self.pose_score_k = float(os.getenv("POSE_SCORE_K", "1.0"))
        # Angle-based check: base angle (radians) inside which rotation is considered normal
        # tuned default: larger angular tolerance (~25 degrees)
        self.pose_angle_base = float(os.getenv("POSE_ANGLE_BASE_RAD", "0.45"))
        # Weight of angular excess when combining with distance excess
        self.pose_angle_weight = float(os.getenv("POSE_ANGLE_WEIGHT", "1.0"))
        # How to combine distance and angle: 'sum' or 'max'
        # tuned default: use 'max' to avoid summing small deviations into false positives
        self.pose_combine_mode = str(os.getenv("POSE_COMBINE_MODE", "max")).lower()
        # Debugging: write per-frame pose values to CSV for tuning (set POSE_DEBUG=1)
        self.pose_debug = str(os.getenv("POSE_DEBUG", "0")) == "1"
        if self.pose_debug:
            self._pose_debug_csv = os.path.join(self.save_dir, "pose_debug.csv")
            if not os.path.exists(self._pose_debug_csv):
                try:
                    with open(self._pose_debug_csv, "w") as fh:
                        fh.write("ts,cid,person_key,dist,dist_excess,abs_angle,angle_excess_norm,combined_excess,score_s\n")
                except Exception:
                    pass
        # Optionally save candidate frames/clips for offline labelling (set POSE_SAVE_CLIPS=1)
        self.pose_save_clips = str(os.getenv("POSE_SAVE_CLIPS", "0")) == "1"
        if self.pose_save_clips:
            self._clip_debug_dir = os.path.join(self.save_dir, "clip_debug")
            os.makedirs(self._clip_debug_dir, exist_ok=True)
        self.detections = [] # Used for table/UI output
        # Pose-based anti-cheating state (per camera)
        self.pose_state = {}  # cam_id_track_id -> state dict
        # Confirmed cheaters (hard stop for this exact person_key)
        self.warned_persons = set()  # person_key

        # Confirmation workflow:
        # - After first warning for a person, they become "pending" until an operator confirms.
        # - If confirmed cheating -> stop tracking (banned).
        # - If false positive -> cooldown 2 minutes, then resume tracking and allow a new warning.
        self.decisions_key = "proctor_decisions"
        self.confirm_cooldown_s = 120.0
        self.pending_ttl_s = 60.0 * 60.0  # auto-expire pending if UI never answers
        # person_key -> {state: pending|cooldown|banned, until: ts, cam_id, track_id}
        self.person_gate: dict[str, dict] = {}

        # Anti-spam: suppress repeats when ByteTrack re-assigns track_id.
        # Storage format: cam_id -> list[{'box': (x1,y1,x2,y2), 'cx': float, 'cy': float, 'state': str, 'until': float}]
        self.warned_boxes_by_cam: dict[str, list[dict]] = {}
        self.warned_box_ttl_s = 24 * 60 * 60  # keep banned zones for 24 hours (session-level)
        self.warned_iou_thr = float(os.getenv("POSE_WARN_IOU_THR", "0.25"))  # lower => more dedupe
        # Pixel distance between bbox centers to treat as the same seated person (more stable than IoU)
        self.pose_warn_center_px = float(os.getenv("POSE_WARN_CENTER_PX", "140"))
        self.start_time = time.time()  # Engine start time for video timeline calculations

        # Clear Redis queues on startup to avoid stale warnings/results
        try:
            self.r.delete("proctor_warnings")
            self.r.delete("raw_ai_results")
            self.r.delete(self.decisions_key)
            print("[INFO] Redis queues cleared at startup")
        except Exception as e:
            print(f"[WARN] Failed to clear Redis queues: {e}")

        # Auto-tune parameters from debug CSV if requested (no manual labelling needed)
        if str(os.getenv("POSE_AUTO_TUNE", "0")) == "1":
            try:
                res = self.auto_tune_from_debug_csv()
                print(f"[AUTO_TUNE] Completed: {res}")
            except Exception as e:
                print(f"[AUTO_TUNE] Failed: {e}")

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

    def _save_evidence_frame(self, frame: np.ndarray, person_box,
                             cam_id: str, track_id: int, video_time_str: str,
                             real_time_str: str, now_ts: float):
        """
        Save evidence frame with cheater's bounding box highlighted.
        Used to visually identify who the cheating person is.
        """
        try:
            # Debug: log frame status
            if frame is None:
                print(f"[EVIDENCE] WARNING: frame is None")
                return None

            if not isinstance(frame, np.ndarray):
                print(f"[EVIDENCE] WARNING: frame is not ndarray, got {type(frame)}")
                return None

            if frame.size == 0:
                print(f"[EVIDENCE] WARNING: frame is empty")
                return None

            print(f"[EVIDENCE] Processing frame: shape={frame.shape}, dtype={frame.dtype}")

            frame_copy = frame.copy()

            # Draw bounding box around the cheating person
            if person_box is not None and len(person_box) >= 4:
                x1, y1, x2, y2 = int(person_box[0]), int(person_box[1]), int(person_box[2]), int(person_box[3])
                # Bright red for the cheater
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
                # Write track_id above the box
                cv2.putText(frame_copy, f"ID:{track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            # Add metadata overlay
            cv2.putText(frame_copy, f"Camera: {cam_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_copy, f"Video time: {video_time_str}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_copy, f"Real time: {real_time_str}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_copy, "CHEATING DETECTED!", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Ensure save directory exists
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                print(f"[EVIDENCE] Created directory: {self.save_dir}")

            # Create evidence filename: cam_id_track_id_timestamp.jpg
            timestamp_ms = int(now_ts * 1000) % 1000000
            evidence_filename = f"cheater_cam{cam_id}_id{track_id}_{timestamp_ms}.jpg"
            evidence_path = os.path.join(self.save_dir, evidence_filename)

            # Try to write file
            success = cv2.imwrite(evidence_path, frame_copy)
            if success:
                print(f"[EVIDENCE] ✅ Saved: {evidence_path}")
            else:
                print(f"[EVIDENCE] ❌ Failed to save: {evidence_path}")
                return None

            # Also save metadata JSON alongside the image
            metadata = {
                "evidence_file": evidence_filename,
                "evidence_path": evidence_path,
                "cam_id": cam_id,
                "track_id": track_id,
                "video_time": video_time_str,
                "real_time": real_time_str,
                "timestamp": now_ts
            }
            metadata_path = evidence_path.replace(".jpg", ".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[EVIDENCE] ✅ Metadata: {metadata_path}")

            return {
                "evidence_file": evidence_filename,
                "evidence_path": evidence_path,
                "metadata_path": metadata_path,
            }

        except Exception as e:
            print(f"[ERROR] Exception in _save_evidence_frame: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

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

    @staticmethod
    def _pose_suspicion_with_angle(person_kpts: np.ndarray):
        """
        Extended pose suspicion: also compute nose-vs-torso signed angle (radians).
        Returns: rel_nose_x, rel_nose_y, shoulder_w, angle_diff, confidence_ok
        angle_diff is in (-pi, pi]
        """
        if person_kpts is None or person_kpts.size == 0:
            return None, None, None, None, False

        def kp(i):
            x, y, c = person_kpts[i]
            return float(x), float(y), float(c)

        nx, ny, nc = kp(0)  # nose
        lsx, lsy, lsc = kp(5)  # left shoulder
        rsx, rsy, rsc = kp(6)  # right shoulder

        # If core keypoints are missing, we cannot compute a reliable position
        if nc < 0.3 or lsc < 0.3 or rsc < 0.3:
            return None, None, None, None, False

        shoulder_w = abs(rsx - lsx)
        if shoulder_w < 1.0:
            return None, None, None, None, False

        mid_x = (lsx + rsx) / 2.0
        mid_y = (lsy + rsy) / 2.0

        rel_nose_x = (nx - mid_x) / shoulder_w
        rel_nose_y = (ny - mid_y) / shoulder_w

        # Torso vector (right - left)
        tx = rsx - lsx
        ty = rsy - lsy
        # Approximate forward vector by rotating torso vector by +90 degrees
        fx = -ty
        fy = tx

        # Compute angles
        try:
            angle_nose = math.atan2(ny - mid_y, nx - mid_x)
            angle_torso = math.atan2(fy, fx)
            angle_diff = (angle_nose - angle_torso + math.pi) % (2 * math.pi) - math.pi
        except Exception:
            angle_diff = 0.0

        return rel_nose_x, rel_nose_y, shoulder_w, float(angle_diff), True

    @staticmethod
    def _bbox_iou_xyxy(a_xyxy, b_xyxy) -> float:
        """IoU for boxes in (x1, y1, x2, y2) format."""
        try:
            ax1, ay1, ax2, ay2 = map(float, a_xyxy)
            bx1, by1, bx2, by2 = map(float, b_xyxy)
        except Exception:
            return 0.0

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = a_area + b_area - inter
        if denom <= 0.0:
            return 0.0
        return float(inter / denom)

    def _cleanup_warned_boxes(self, cam_id: str, now_ts: float) -> None:
        cam_id = str(cam_id)
        lst = self.warned_boxes_by_cam.get(cam_id)
        if not lst:
            return
        alive = [it for it in lst if float(now_ts) <= float(it.get("until", 0.0))]
        if len(alive) != len(lst):
            self.warned_boxes_by_cam[cam_id] = alive

    def _match_warned_box(self, cam_id: str, person_box, now_ts: float) -> dict | None:
        """Return matched warned-box record {state, until, box} or None."""
        if person_box is None or len(person_box) < 4:
            return None

        cam_id = str(cam_id)
        self._cleanup_warned_boxes(cam_id, now_ts)
        lst = self.warned_boxes_by_cam.get(cam_id, [])
        if not lst:
            return None

        x1, y1, x2, y2 = float(person_box[0]), float(person_box[1]), float(person_box[2]), float(person_box[3])
        cur = (x1, y1, x2, y2)
        cur_cx = (x1 + x2) / 2.0
        cur_cy = (y1 + y2) / 2.0

        for it in lst:
            box = it.get("box")
            if not box:
                continue
            # Match either by IoU OR by stable center distance (handles leaning / bbox resize)
            iou_ok = self._bbox_iou_xyxy(cur, box) >= float(self.warned_iou_thr)
            try:
                it_cx = float(it.get("cx"))
                it_cy = float(it.get("cy"))
                d = ((cur_cx - it_cx) ** 2 + (cur_cy - it_cy) ** 2) ** 0.5
                center_ok = d <= float(self.pose_warn_center_px)
            except Exception:
                center_ok = False

            if (iou_ok or center_ok) and float(now_ts) <= float(it.get("until", 0.0)):
                return it
        return None

    def _upsert_warned_box(self, cam_id: str, person_box, now_ts: float,
                           state: str, until: float) -> None:
        """Insert or update warned-box record for this camera by IoU match."""
        if person_box is None or len(person_box) < 4:
            return
        cam_id = str(cam_id)
        self._cleanup_warned_boxes(cam_id, now_ts)

        x1, y1, x2, y2 = float(person_box[0]), float(person_box[1]), float(person_box[2]), float(person_box[3])
        cur = (x1, y1, x2, y2)
        cur_cx = (x1 + x2) / 2.0
        cur_cy = (y1 + y2) / 2.0
        lst = self.warned_boxes_by_cam.setdefault(cam_id, [])

        for it in lst:
            box = it.get("box")
            if not box:
                continue
            # Update if same by IoU OR by center distance
            iou_ok = self._bbox_iou_xyxy(cur, box) >= float(self.warned_iou_thr)
            try:
                it_cx = float(it.get("cx"))
                it_cy = float(it.get("cy"))
                d = ((cur_cx - it_cx) ** 2 + (cur_cy - it_cy) ** 2) ** 0.5
                center_ok = d <= float(self.pose_warn_center_px)
            except Exception:
                center_ok = False

            if iou_ok or center_ok:
                it["state"] = str(state)
                it["until"] = float(until)
                it["box"] = cur
                it["cx"] = float(cur_cx)
                it["cy"] = float(cur_cy)
                return

        lst.append({
            "box": cur,
            "cx": float(cur_cx),
            "cy": float(cur_cy),
            "state": str(state),
            "until": float(until),
        })

    def _drain_decisions(self, now_ts: float, max_items: int = 50) -> None:
        """Apply operator decisions coming from the warnings UI."""
        for _ in range(int(max_items)):
            try:
                payload = self.r.lpop(self.decisions_key)
            except Exception:
                payload = None

            if not payload:
                break

            try:
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8", errors="replace")
                obj = json.loads(payload)
            except Exception:
                continue

            cam_id = str(obj.get("cam_id", "unknown"))
            track_id = int(obj.get("track_id", 0) or 0)
            person_key = str(obj.get("person_key") or f"{cam_id}_{track_id}")
            decision = str(obj.get("decision", "")).strip().lower()

            prev_gate = self.person_gate.get(person_key) or {}
            prev_box = prev_gate.get("box")
            if prev_box is None:
                st_prev = self.pose_state.get(person_key)
                if st_prev is not None:
                    prev_box = st_prev.get("_last_person_box")

            # If UI sends decision for an unknown person, we still store a gate to suppress spam.
            if decision in ("cheating", "yes", "true", "1"):
                # Hard stop
                self.warned_persons.add(person_key)
                self.person_gate[person_key] = {
                    "state": "banned",
                    "until": float(now_ts) + float(self.warned_box_ttl_s),
                    "cam_id": cam_id,
                    "track_id": track_id,
                    "box": prev_box,
                }
                if prev_box is not None:
                    self._upsert_warned_box(cam_id, prev_box, now_ts, state="banned", until=float(now_ts) + float(self.warned_box_ttl_s))

            elif decision in ("not_cheating", "false", "no", "0"):
                # Cooldown then resume tracking
                until = float(now_ts) + float(self.confirm_cooldown_s)
                self.person_gate[person_key] = {
                    "state": "cooldown",
                    "until": until,
                    "cam_id": cam_id,
                    "track_id": track_id,
                    "box": prev_box,
                }
                # Reset pose state so we don't instantly re-trigger after cooldown
                try:
                    self.pose_state.pop(person_key, None)
                except Exception:
                    pass
                if prev_box is not None:
                    self._upsert_warned_box(cam_id, prev_box, now_ts, state="cooldown", until=until)

            # else: unknown decision -> ignore

    def _update_pose_warning_for_person(self, person_key: str, cam_id: str, track_id: int,
                                          person_kpts: np.ndarray, person_box, now_ts: float,
                                          current_frame: np.ndarray = None):
        """
        Process one tracked person by track_id.
        person_key: unique key in format "cam_id_track_id"
        person_box: [x1, y1, x2, y2, conf, cls]
        """
        # Hard stop: confirmed cheater
        if person_key in self.warned_persons:
            return ""

        # Gate by confirmation workflow state (pending/cooldown/banned)
        gate = self.person_gate.get(person_key)
        if gate is not None:
            state = str(gate.get("state", ""))
            until = float(gate.get("until", 0.0) or 0.0)
            if state == "pending":
                # Wait for operator, but auto-release if UI never answers
                if float(now_ts) < until:
                    return ""
                try:
                    del self.person_gate[person_key]
                except Exception:
                    pass
            if state == "banned":
                return ""
            if state == "cooldown":
                if float(now_ts) < until:
                    return ""
                # Cooldown is over -> resume
                try:
                    del self.person_gate[person_key]
                except Exception:
                    pass

        # If tracker reassigned track_id, suppress by bbox match as well
        match = self._match_warned_box(cam_id, person_box, now_ts)
        if match is not None:
            m_state = str(match.get("state", ""))
            m_until = float(match.get("until", 0.0) or 0.0)
            if float(now_ts) <= m_until and m_state in ("pending", "cooldown", "banned"):
                self.person_gate[person_key] = {
                    "state": m_state,
                    "until": m_until,
                    "cam_id": str(cam_id),
                    "track_id": int(track_id),
                    "box": match.get("box"),
                }
                return ""

        st = self.pose_state.get(person_key)
        if st is None:
            st = {
                "score_s": 0.0,           # Accumulated "suspicion seconds"
                "last_warn_ts": 0.0,
                "last_ts": now_ts,
                "calib_end_ts": now_ts + float(self.pose_calib_s),  # Auto-calibration window
                "base_nose_x": None,      # Baseline nose X position (circle center)
                "base_nose_y": None,      # Baseline nose Y position (circle center)
                "base_radius": float(self.pose_base_radius),      # Radius of the "normal zone" (smaller = more sensitive)
                "max_away_dist": float(self.pose_max_away_dist),  # Threshold for "person moved far away" (do not warn)
                "is_away": False,         # Flag indicating the person is far away
                "track_id": track_id,
                # Walking detection fields
                "last_box_cx": None,      # Previous bbox center X
                "last_box_cy": None,      # Previous bbox center Y
                "walk_speed_threshold": float(self.pose_walk_speed_threshold),  # Pixels per frame threshold for walking
            }

        # Remember last bbox for confirmation decisions
        st["_last_person_box"] = person_box

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

        # Compute nose position and relative angle for this person
        rel_nose_x, rel_nose_y, shoulder_w, angle_diff = None, None, None, None
        try:
            rel_nose_x, rel_nose_y, shoulder_w, angle_diff, conf_ok = self._pose_suspicion_with_angle(person_kpts)
            if not conf_ok:
                # Not suspicious: skip this frame when keypoint confidence is insufficient
                rel_nose_x, rel_nose_y, angle_diff = None, None, None
        except Exception:
            # Not suspicious: skip this frame on pose parsing errors
            rel_nose_x, rel_nose_y, angle_diff = None, None, None

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

        # combined_excess aggregates distance and angular deviations (initialized to zero)
        combined_excess = 0.0

        if rel_nose_x is not None and rel_nose_y is not None:
            if st["base_nose_x"] is not None and st["base_nose_y"] is not None:
                # Distance from baseline center to current nose position (normalized units)
                dist = ((rel_nose_x - st["base_nose_x"])**2 + (rel_nose_y - st["base_nose_y"])**2) ** 0.5

                # Compute angular excess (normalized by angle base)
                angle_excess_norm = 0.0
                if angle_diff is not None:
                    abs_angle = abs(float(angle_diff))
                    angle_excess = max(0.0, abs_angle - float(self.pose_angle_base))
                    angle_excess_norm = angle_excess / (float(self.pose_angle_base) + 1e-6)

                # Combine distance excess and angular excess
                dist_excess = max(0.0, dist - st["base_radius"]) if dist is not None else 0.0
                # two modes: sum (sensitive) or max (stricter)
                if self.pose_combine_mode == "max":
                    combined_excess = max(float(dist_excess), float(self.pose_angle_weight) * float(angle_excess_norm))
                else:
                    combined_excess = float(dist_excess) + float(self.pose_angle_weight) * float(angle_excess_norm)

                # Debug: optionally log per-frame pose features to CSV for offline tuning
                if getattr(self, "pose_debug", False):
                    try:
                        ts = now_ts
                        cid = cam_id
                        pk = person_key
                        abs_angle = abs(float(angle_diff)) if angle_diff is not None else 0.0
                        score_now = float(st.get("score_s", 0.0) or 0.0)
                        line = f"{ts},{cid},{pk},{dist:.4f},{dist_excess:.4f},{abs_angle:.4f},{angle_excess_norm:.4f},{combined_excess:.4f},{score_now:.4f}\n"
                        with open(getattr(self, "_pose_debug_csv", os.path.join(self.save_dir, "pose_debug.csv")), "a") as fh:
                            fh.write(line)
                    except Exception:
                        pass

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
                # Regular out-of-circle or angular behavior triggers warning accumulation
                # Calibration must be completed first
                elif combined_excess > 0.0 and not st["is_away"] and not in_calibration:
                    suspicious = True

                # Optionally save candidate frame for offline labelling (throttle per person)
                if suspicious and getattr(self, "pose_save_clips", False) and current_frame is not None:
                    last_ts = float(st.get("_last_saved_clip_ts", 0.0) or 0.0)
                    # throttle saving: at most one per 5 seconds per person
                    if float(now_ts) - last_ts > 5.0:
                        try:
                            fname = f"candidate_cam{cam_id}_id{track_id}_{int(now_ts)}.jpg"
                            fpath = os.path.join(self._clip_debug_dir, fname)
                            cv2.imwrite(fpath, current_frame)
                            st["_last_saved_clip_ts"] = float(now_ts)
                        except Exception:
                            pass

                # DEBUG: print state every 30 frames
                if self.frameCount % 30 == 0:
                    status = "CALIB" if in_calibration else ("SUSP!" if suspicious else "ok")
                    print(f"[POSE] {person_key}: dist={dist:.2f} radius={st['base_radius']:.2f} [{status}]")

        # Time-based accumulation
        if is_walking:
            # Person is walking: decay score faster
            st["score_s"] = max(0.0, float(st["score_s"]) - dt * 2.0)
        elif suspicious:
            # Accumulate score proportional to combined_excess (stronger deviations -> faster accumulation)
            try:
                add = float(self.pose_score_k) * float(combined_excess) * dt
            except Exception:
                add = dt * 1.0
            st["score_s"] = min(15.0, float(st["score_s"]) + add)
        else:
            # Decay score when behavior returns to normal
            st["score_s"] = max(0.0, float(st["score_s"]) - dt * 1.0)

        # DEBUG: print score roughly once per second
        if suspicious and int(st["score_s"]) != int(st.get("_last_print_score", -1)):
            st["_last_print_score"] = int(st["score_s"])
            print(f"[DEBUG] {person_key}: score={st['score_s']:.1f}/{self.threshold} (suspicious={suspicious})")

        # Emit warning once per person until operator confirms
        if float(st["score_s"]) >= float(self.threshold):
            # If this physical person is already pending/banned/cooldown by bbox, do nothing
            match2 = self._match_warned_box(cam_id, person_box, now_ts)
            if match2 is not None:
                m_state = str(match2.get("state", ""))
                m_until = float(match2.get("until", 0.0) or 0.0)
                if float(now_ts) <= m_until and m_state in ("pending", "cooldown", "banned"):
                    self.person_gate[person_key] = {
                        "state": m_state,
                        "until": m_until,
                        "cam_id": str(cam_id),
                        "track_id": int(track_id),
                        "box": match2.get("box"),
                    }
                    self.pose_state[person_key] = st
                    return ""

            warn_text = "dist_warning"

            # Put this person into PENDING state immediately to stop spam
            pending_until = float(now_ts) + float(self.pending_ttl_s)
            self.person_gate[person_key] = {
                "state": "pending",
                "until": pending_until,
                "cam_id": str(cam_id),
                "track_id": int(track_id),
                "box": person_box,
            }
            self._upsert_warned_box(cam_id, person_box, now_ts, state="pending", until=pending_until)

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

            warning_data: dict[str, object] = {
                "ts": now_ts,
                "cam_id": cam_id,
                "track_id": track_id,
                "person_key": person_key,
                "type": "dist_warning",
                "video_time": video_time_str,
                "real_time": current_time_str
            }
            # Include bbox for easier debugging and possible future UI highlight
            try:
                if person_box is not None and len(person_box) >= 4:
                    warning_data["box"] = [
                        int(float(person_box[0])),
                        int(float(person_box[1])),
                        int(float(person_box[2])),
                        int(float(person_box[3])),
                    ]
            except Exception:
                pass
            # Save evidence frame with bounding box and metadata
            evidence = self._save_evidence_frame(
                frame=current_frame,
                person_box=person_box,
                cam_id=cam_id,
                track_id=track_id,
                video_time_str=video_time_str,
                real_time_str=current_time_str,
                now_ts=now_ts
            )

            if isinstance(evidence, dict):
                warning_data.update({
                    "evidence_file": evidence.get("evidence_file"),
                    "evidence_path": evidence.get("evidence_path"),
                    "metadata_path": evidence.get("metadata_path"),
                })

            print(f"[DEBUG] Sending to Redis: {warning_data}")
            try:
                self.r.rpush("proctor_warnings", json.dumps(warning_data))
            except Exception as e:
                print(f"[ERROR] Redis error: {e}")

        self.pose_state[person_key] = st
        return warn_text

    def _update_pose_warning(self, cam_id: str, boxes: np.ndarray, kpts: np.ndarray,
                              track_ids: np.ndarray, now_ts: float, current_frame: np.ndarray = None):
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
            person_box = None
            if boxes is not None and idx < len(boxes):
                try:
                    b = boxes[idx]
                    # Keep only xyxy in a simple tuple so it survives across frames/decisions
                    person_box = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                except Exception:
                    person_box = None

            warn_text = self._update_pose_warning_for_person(
                person_key, cam_id, track_id, person_kpts, person_box, now_ts, current_frame
            )

            # Only append dist_warning events
            if warn_text == "dist_warning":
                warnings.append({"track_id": track_id, "type": "dist_warning"})
                print(f"[ENGINE] ✅ Warning added to list: track_id={track_id}")

        return warnings

    def auto_tune_from_debug_csv(self, csv_path: str = None, target_fah: float = 0.5,
                                 decay: float = 1.0, cooldown_s: float = 60.0,
                                 k_grid: list | None = None, percentile: float = 95.0):
        """
        Auto-tune base_radius, angle_base and pose_score_k from a previously collected
        debug CSV (written when POSE_DEBUG=1). This is unsupervised: it assumes the
        CSV mostly contains 'normal' behavior and chooses parameters so that the
        expected false alarms per hour (FAH) on this data is <= target_fah.

        Returns a dict with chosen parameters and statistics.
        """
        csv_path = csv_path or getattr(self, "_pose_debug_csv", os.path.join(self.save_dir, "pose_debug.csv"))
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Debug CSV not found: {csv_path}")

        rows = []
        with open(csv_path, "r") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                try:
                    ts = float(r.get("ts", 0.0))
                    pk = r.get("person_key", "unknown")
                    dist = float(r.get("dist", 0.0))
                    abs_angle = float(r.get("abs_angle", 0.0)) if r.get("abs_angle") is not None else 0.0
                    rows.append({"ts": ts, "person_key": pk, "dist": dist, "abs_angle": abs_angle})
                except Exception:
                    continue

        if len(rows) == 0:
            raise ValueError("No usable rows in debug CSV")

        # compute suggested base_radius and angle_base as high-percentiles of observed values
        dists = np.array([r["dist"] for r in rows], dtype=float)
        angles = np.array([r["abs_angle"] for r in rows], dtype=float)
        base_radius_sugg = float(np.percentile(dists, percentile))
        angle_base_sugg = float(np.percentile(angles, percentile))

        # prepare combined_excess per row for candidate parameters
        for r in rows:
            # will compute later per candidate
            r["dist"] = float(r["dist"])
            r["abs_angle"] = float(r["abs_angle"])

        # total hours in dataset
        ts_vals = [r["ts"] for r in rows]
        total_hours = max(1e-6, (max(ts_vals) - min(ts_vals))) / 3600.0

        # grid for k if not provided
        if k_grid is None:
            k_grid = list(np.linspace(0.1, 3.0, 30))

        def simulate_for_k(k_val):
            # simulate per-person score progression and count alarms
            alarms = 0
            rows_by_person = {}
            for r in rows:
                rows_by_person.setdefault(r["person_key"], []).append(r)

            for pk, lst in rows_by_person.items():
                lst_sorted = sorted(lst, key=lambda x: x["ts"])
                score = 0.0
                last_ts = lst_sorted[0]["ts"]
                skip_until = -1.0
                for rec in lst_sorted:
                    ts = rec["ts"]
                    if ts < skip_until:
                        last_ts = ts
                        continue
                    dt = max(0.0, min(0.25, ts - last_ts))
                    last_ts = ts
                    # compute new combined_excess with suggested bases
                    dist_excess = max(0.0, rec["dist"] - base_radius_sugg)
                    angle_excess = max(0.0, rec["abs_angle"] - angle_base_sugg)
                    angle_excess_norm = angle_excess / (angle_base_sugg + 1e-6)
                    if self.pose_combine_mode == "max":
                        combined = max(dist_excess, float(self.pose_angle_weight) * angle_excess_norm)
                    else:
                        combined = dist_excess + float(self.pose_angle_weight) * angle_excess_norm

                    if combined > 0.0:
                        score = min(1e6, score + k_val * combined * dt)
                    else:
                        score = max(0.0, score - decay * dt)

                    if score >= float(self.threshold):
                        alarms += 1
                        score = 0.0
                        skip_until = ts + float(cooldown_s)

            fah = alarms / max(1e-9, total_hours)
            return fah

        # search best k: maximize sensitivity while fah <= target_fah
        best_k = None
        best_fah = None
        for k_candidate in k_grid:
            fah = simulate_for_k(k_candidate)
            if best_k is None:
                best_k = k_candidate
                best_fah = fah
            else:
                # prefer larger k if still under target_fah
                if fah <= target_fah and (best_fah is None or best_fah > target_fah or k_candidate > best_k):
                    best_k = k_candidate
                    best_fah = fah

        # set tuned parameters
        prev_params = {"pose_base_radius": self.pose_base_radius,
                       "pose_angle_base": self.pose_angle_base,
                       "pose_score_k": self.pose_score_k}

        self.pose_base_radius = base_radius_sugg
        self.pose_angle_base = angle_base_sugg
        self.pose_score_k = float(best_k)

        result = {
            "base_radius_sugg": base_radius_sugg,
            "angle_base_sugg": angle_base_sugg,
            "chosen_k": float(best_k),
            "chosen_fah": float(best_fah),
            "prev_params": prev_params,
            "total_hours": total_hours,
            "rows_used": len(rows),
        }

        # persist tuning to a small json file for record
        try:
            outp = os.path.join(self.save_dir, "auto_tune_result.json")
            with open(outp, "w") as fh:
                json.dump(result, fh, indent=2)
        except Exception:
            pass

        return result

    def process_batch(self, frames, cam_ids):
        if not frames or any(f is None for f in frames):
            return frames, self.detections

        current_time = time.time()
        # Apply operator confirmations ASAP (cheap non-blocking lpop loop)
        try:
            self._drain_decisions(current_time)
        except Exception:
            pass
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
                imgsz=1280,
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
            warnings = self._update_pose_warning(str(cam_id), boxes, kpts, track_ids, current_time, frames[i])

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

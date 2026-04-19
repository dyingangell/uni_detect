import json
import os
import queue
import threading
import time
from dataclasses import dataclass

import base64

import redis
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2


REDIS_HOST = "localhost"
REDIS_PORT = 6379
WARNINGS_KEY = "proctor_warnings"
DECISIONS_KEY = "proctor_decisions"


@dataclass
class WarningEvent:
    ts: float
    cam_id: str
    type: str
    track_id: int = 0
    video_time: str = ""
    real_time: str = ""
    person_key: str = ""
    status: str = "pending"  # pending|cheating|not_cheating    ffff
    decision_ts: float = 0.0
    box: list[int] | None = None
    evidence_path: str = ""
    evidence_file: str = ""

    @staticmethod
    def from_payload(payload: str) -> "WarningEvent | None":
        try:
            obj = json.loads(payload)
            print(f"[WARN_WINDOW] Received from Redis: {obj}")
            wtype = str(obj.get("type", ""))
            # Accept only dist_warning events
            if wtype != "dist_warning":
                print(f"[WARN_WINDOW] Skipping type: {wtype}")
                return None
            ts = float(obj.get("ts", time.time()))
            cam_id = str(obj.get("cam_id", "unknown"))
            track_id = int(obj.get("track_id", 0))
            video_time = str(obj.get("video_time", ""))
            real_time = str(obj.get("real_time", ""))
            person_key = str(obj.get("person_key") or f"{cam_id}_{track_id}")
            box = obj.get("box")
            evidence_path = str(obj.get("evidence_path") or "")
            evidence_file = str(obj.get("evidence_file") or "")
            if isinstance(box, list) and len(box) == 4:
                try:
                    box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                except Exception:
                    box = None
            else:
                box = None
            print(f"[WARN_WINDOW] ✅ Accepted dist_warning for camera {cam_id}, track_id={track_id}, video_time={video_time}")
            return WarningEvent(
                ts=ts,
                cam_id=cam_id,
                type=wtype,
                track_id=track_id,
                video_time=video_time,
                real_time=real_time,
                person_key=person_key,
                status="pending",
                box=box,
                evidence_path=evidence_path,
                evidence_file=evidence_file,
            )
        except Exception as e:
            print(f"[WARN_WINDOW] Parsing error: {e}")
            return None

    def pretty(self) -> str:
        tstr = time.strftime("%H:%M:%S", time.localtime(self.ts))
        st = (self.status or "pending").upper()
        if st == "PENDING":
            tag = "PENDING"
        elif st in ("CHEATING", "YES", "TRUE"):
            tag = "CHEATING"
        elif st in ("NOT_CHEATING", "FALSE", "NO"):
            tag = "FALSE"
        else:
            tag = st
        return f"[{tstr}] cam={self.cam_id} id={self.track_id} {self.type} [{tag}]"


class WarningsApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Proctoring warnings")
        self.root.geometry("680x420")

        self.events: list[WarningEvent] = []
        self.ui_queue: "queue.Queue[WarningEvent]" = queue.Queue()
        self.stop_evt = threading.Event()

        # Avoid re-opening the review window repeatedly on UI refreshes
        self._last_review_key: tuple[str, float] | None = None

        # Redis client for sending operator decisions
        self.sender_redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

        # top controls
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Filter cam_id:").pack(side="left")
        self.filter_var = tk.StringVar(value="")
        self.filter_entry = ttk.Entry(top, textvariable=self.filter_var, width=12)
        self.filter_entry.pack(side="left", padx=(6, 10))
        self.filter_entry.bind("<KeyRelease>", lambda _e: self._refresh_list())

        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Auto-scroll", variable=self.auto_scroll_var).pack(side="left")

        ttk.Button(top, text="Show cheaters", command=self._show_cheaters).pack(side="left", padx=(10, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(top, text="Review…", command=self._open_review_for_selected).pack(side="left")

        ttk.Button(top, text="Clear", command=self._clear).pack(side="right")
        ttk.Button(top, text="Export", command=self._export).pack(side="right", padx=(0, 8))

        # main list + scrollbar
        body = ttk.Frame(root, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)

        self.listbox = tk.Listbox(body, activestyle="none")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        self.listbox.bind("<Double-Button-1>", lambda _e: self._open_review_for_selected())
        # Single-click UX: click a pending warning -> open review with photo
        self.listbox.bind("<ButtonRelease-1>", lambda _e: self.root.after(10, self._maybe_open_review_on_click))

        sb = ttk.Scrollbar(body, orient="vertical", command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=sb.set)

        # details panel
        self.details = tk.Text(root, height=5, wrap="word")
        self.details.pack(fill="x", padx=8, pady=(0, 8))
        self.details.configure(state="disabled")

        # background reader
        self.reader_thread = threading.Thread(target=self._redis_reader, daemon=True)
        self.reader_thread.start()

        # UI polling
        self.root.after(50, self._drain_ui_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _redis_reader(self):
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        try:
            r.ping()
        except Exception as e:
            self.ui_queue.put(WarningEvent(ts=time.time(), cam_id="system", type=f"redis_down:{e}"))
            return

        while not self.stop_evt.is_set():
            try:
                item = r.blpop(WARNINGS_KEY, timeout=1)
                if not item:
                    continue
                _key, payload = item
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8", errors="replace")
                ev = WarningEvent.from_payload(payload)
                if ev is not None:
                    self.ui_queue.put(ev)
            except Exception as e:
                self.ui_queue.put(WarningEvent(ts=time.time(), cam_id="system", type=f"redis_err:{e}"))
                time.sleep(0.5)

    def _drain_ui_queue(self):
        changed = False
        try:
            while True:
                ev = self.ui_queue.get_nowait()
                self.events.append(ev)
                changed = True
        except queue.Empty:
            pass

        if changed:
            self._refresh_list()
            if self.auto_scroll_var.get():
                self.listbox.yview_moveto(1.0)

        self.root.after(80, self._drain_ui_queue)

    def _filtered_indices(self) -> list[int]:
        f = self.filter_var.get().strip()
        if not f:
            return list(range(len(self.events)))
        return [i for i, ev in enumerate(self.events) if ev.cam_id == f]

    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i in self._filtered_indices():
            self.listbox.insert(tk.END, self.events[i].pretty())

    def _on_select(self, _evt=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        visible_idx = int(sel[0])
        real_indices = self._filtered_indices()
        if visible_idx >= len(real_indices):
            return
        ev = self.events[real_indices[visible_idx]]
        self.details.configure(state="normal")
        self.details.delete("1.0", tk.END)
        self.details.insert("1.0", json.dumps({
            "ts": ev.ts,
            "cam_id": ev.cam_id,
            "track_id": ev.track_id,
            "person_key": ev.person_key,
            "type": ev.type,
            "video_time": ev.video_time,
            "real_time": ev.real_time,
            "status": ev.status,
            "decision_ts": ev.decision_ts,
            "box": ev.box,
            "evidence_path": ev.evidence_path,
            "evidence_file": ev.evidence_file,
        }, ensure_ascii=False, indent=2))
        self.details.configure(state="disabled")

    def _get_selected_event(self) -> tuple[int, WarningEvent] | None:
        sel = self.listbox.curselection()
        if not sel:
            return None
        visible_idx = int(sel[0])
        real_indices = self._filtered_indices()
        if visible_idx >= len(real_indices):
            return None
        real_idx = real_indices[visible_idx]
        return real_idx, self.events[real_idx]

    def _confirm_event(self, real_idx: int, ev: WarningEvent, decision: str):
        if ev.cam_id == "system":
            return
        if ev.status != "pending":
            messagebox.showinfo("Confirm", f"Already decided: {ev.status}")
            return

        payload = {
            "ts": time.time(),
            "cam_id": ev.cam_id,
            "track_id": ev.track_id,
            "person_key": ev.person_key or f"{ev.cam_id}_{ev.track_id}",
            "decision": str(decision),
        }

        try:
            self.sender_redis.rpush(DECISIONS_KEY, json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            messagebox.showerror("Confirm failed", str(e))
            return

        # Update local UI state
        ev.status = "cheating" if decision == "cheating" else "not_cheating"
        ev.decision_ts = payload["ts"]
        self.events[real_idx] = ev
        self._refresh_list()
        self._on_select()

    def _confirm_selected(self, decision: str):
        picked = self._get_selected_event()
        if picked is None:
            messagebox.showinfo("Confirm", "Select an event first.")
            return
        real_idx, ev = picked
        self._confirm_event(real_idx, ev, decision)

    def _open_review_for_selected(self):
        picked = self._get_selected_event()
        if picked is None:
            messagebox.showinfo("Review", "Select an event first.")
            return
        real_idx, ev = picked
        if ev.cam_id == "system":
            return
        if ev.status != "pending":
            messagebox.showinfo("Review", f"Already decided: {ev.status}")
            return
        self._open_review_window(real_idx, ev)

    def _maybe_open_review_on_click(self):
        picked = self._get_selected_event()
        if picked is None:
            return
        real_idx, ev = picked
        if ev.cam_id == "system":
            return
        if ev.status != "pending":
            return
        key = (ev.person_key or f"{ev.cam_id}_{ev.track_id}", float(ev.ts))
        if self._last_review_key == key:
            return
        self._last_review_key = key
        self._open_review_window(real_idx, ev)

    def _open_review_window(self, real_idx: int, ev: WarningEvent):
        win = tk.Toplevel(self.root)
        win.title("Confirm warning")
        win.geometry("520x560")
        win.resizable(False, False)
        try:
            win.transient(self.root)
            win.grab_set()
        except Exception:
            pass

        header = ttk.Frame(win, padding=8)
        header.pack(fill="x")
        ttk.Label(header, text=f"Cam: {ev.cam_id} | ID: {ev.track_id} | {ev.video_time}").pack(side="left")

        img_frame = ttk.Frame(win, padding=(8, 0, 8, 8))
        img_frame.pack(fill="both", expand=True)

        canvas = ttk.Label(img_frame)
        canvas.pack(fill="both", expand=True)

        path = (ev.evidence_path or "").strip()
        if not path and ev.evidence_file:
            path = os.path.join("evidence_folder", ev.evidence_file)

        photo = None
        if path and os.path.isfile(path):
            try:
                img_bgr = cv2.imread(path)
                if img_bgr is not None and img_bgr.size > 0:
                    h, w = img_bgr.shape[:2]
                    max_w, max_h = 480, 420
                    scale = min(max_w / max(1, w), max_h / max(1, h))
                    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                    img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ok, png_buf = cv2.imencode(".png", img_rgb)
                    if ok:
                        b64 = base64.b64encode(png_buf.tobytes()).decode("ascii")
                        photo = tk.PhotoImage(data=b64)
            except Exception:
                photo = None

        if photo is not None:
            canvas.configure(image=photo)
            canvas.image = photo  # keep reference
        else:
            canvas.configure(text="No evidence image found\n" + (path or ""), anchor="center")

        btns = ttk.Frame(win, padding=8)
        btns.pack(fill="x")

        def approve():
            self._confirm_event(real_idx, ev, "cheating")
            try:
                win.destroy()
            except Exception:
                pass

        def reject():
            self._confirm_event(real_idx, ev, "not_cheating")
            try:
                win.destroy()
            except Exception:
                pass

        ttk.Button(btns, text="✅ Approve (списывает)", command=approve).pack(side="left")
        ttk.Button(btns, text="❌ No (ложный)", command=reject).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Close", command=lambda: win.destroy()).pack(side="right")

    def _clear(self):
        if not messagebox.askyesno("Clear warnings", "Clear warnings list in this window?"):
            return
        self.events.clear()
        self._refresh_list()
        self.details.configure(state="normal")
        self.details.delete("1.0", tk.END)
        self.details.configure(state="disabled")

    def _export(self):
        path = filedialog.asksaveasfilename(
            title="Export warnings",
            defaultextension=".jsonl",
            filetypes=[("JSON Lines", "*.jsonl"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                for ev in self.events:
                    warning_data = {
                        "ts": ev.ts,
                        "cam_id": ev.cam_id,
                        "track_id": ev.track_id,
                        "person_key": ev.person_key,
                        "type": ev.type,
                        "video_time": ev.video_time,
                        "real_time": ev.real_time,
                        "status": ev.status,
                        "decision_ts": ev.decision_ts,
                        "box": ev.box,
                        "evidence_path": ev.evidence_path,
                        "evidence_file": ev.evidence_file,
                    }
                    f.write(json.dumps(warning_data, ensure_ascii=False) + "\n")
            messagebox.showinfo("Export", f"Saved {len(self.events)} events to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _on_close(self):
        self.stop_evt.set()
        self.root.destroy()

    def _show_cheaters(self):
        # Unique cam_id list without duplicates: keep only the latest event per camera
        latest_by_cam: dict[str, WarningEvent] = {}
        for ev in self.events:
            if ev.cam_id == "system":
                continue
            prev = latest_by_cam.get(ev.cam_id)
            if prev is None or ev.ts >= prev.ts:
                latest_by_cam[ev.cam_id] = ev

        cams = sorted(latest_by_cam.keys(), key=lambda x: (len(x), x))

        win = tk.Toplevel(self.root)
        win.title("Cheaters (unique cams)")
        win.geometry("620x360")

        header = ttk.Frame(win, padding=8)
        header.pack(fill="x")
        ttk.Label(header, text=f"Unique cams with warnings: {len(cams)}").pack(side="left")

        body = ttk.Frame(win, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)

        lb = tk.Listbox(body, activestyle="none")
        lb.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(body, orient="vertical", command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.configure(yscrollcommand=sb.set)

        for cam_id in cams:
            lb.insert(tk.END, latest_by_cam[cam_id].pretty())

        hint = ttk.Label(win, text="Tip: double-click an entry to filter main list by this cam_id.")
        hint.pack(fill="x", padx=8, pady=(0, 8))

        def on_double(_evt=None):
            sel = lb.curselection()
            if not sel:
                return
            line = lb.get(sel[0])
            # line looks like: "[HH:MM:SS] cam=<id> ..."
            try:
                part = line.split("cam=", 1)[1]
                cam = part.split(" ", 1)[0].strip()
            except Exception:
                return
            self.filter_var.set(cam)
            self._refresh_list()
            if self.auto_scroll_var.get():
                self.listbox.yview_moveto(1.0)

        lb.bind("<Double-Button-1>", on_double)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    app = WarningsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()


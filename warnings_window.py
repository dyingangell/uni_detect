import json
import queue
import threading
import time
from dataclasses import dataclass

import redis
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


REDIS_HOST = "localhost"
REDIS_PORT = 6379
WARNINGS_KEY = "proctor_warnings"


@dataclass
class WarningEvent:
    ts: float
    cam_id: str
    type: str

    @staticmethod
    def from_payload(payload: str) -> "WarningEvent | None":
        try:
            obj = json.loads(payload)
            ts = float(obj.get("ts", time.time()))
            cam_id = str(obj.get("cam_id", "unknown"))
            wtype = str(obj.get("type", "warning"))
            return WarningEvent(ts=ts, cam_id=cam_id, type=wtype)
        except Exception:
            return None

    def pretty(self) -> str:
        tstr = time.strftime("%H:%M:%S", time.localtime(self.ts))
        return f"[{tstr}] cam={self.cam_id} {self.type}"


class WarningsApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Proctoring warnings")
        self.root.geometry("680x420")

        self.events: list[WarningEvent] = []
        self.ui_queue: "queue.Queue[WarningEvent]" = queue.Queue()
        self.stop_evt = threading.Event()

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

        ttk.Button(top, text="Clear", command=self._clear).pack(side="right")
        ttk.Button(top, text="Export", command=self._export).pack(side="right", padx=(0, 8))

        # main list + scrollbar
        body = ttk.Frame(root, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)

        self.listbox = tk.Listbox(body, activestyle="none")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

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
        self.details.insert("1.0", json.dumps({"ts": ev.ts, "cam_id": ev.cam_id, "type": ev.type}, ensure_ascii=False, indent=2))
        self.details.configure(state="disabled")

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
                    f.write(json.dumps({"ts": ev.ts, "cam_id": ev.cam_id, "type": ev.type}, ensure_ascii=False) + "\n")
            messagebox.showinfo("Export", f"Saved {len(self.events)} events to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _on_close(self):
        self.stop_evt.set()
        self.root.destroy()

    def _show_cheaters(self):
        # Уникальные cam_id без повторов: берём последнее событие на камеру
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


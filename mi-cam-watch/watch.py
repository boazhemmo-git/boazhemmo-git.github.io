#!/usr/bin/env python3
"""mi-cam-watch: alert on people and animals, ignore wind.

    python watch.py --config config.toml

Reads camera streams (RTSP from go2rtc, or video files for testing), saves an
annotated snapshot per event, optionally pushes a notification, and serves a
small local dashboard.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import threading
import time
import tomllib
import urllib.parse
import urllib.request
import uuid
from dataclasses import fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
# Camera streams often start mid-GOP; FFmpeg then prints a warning per frame until
# the next keyframe. Those drown out our own output, so silence FFmpeg.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2  # noqa: E402

from detector import YoloxDetector, resolve_model  # noqa: E402
from pipeline import Pipeline, Settings  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_CLASSES = ["person", "dog", "cat", "bird", "horse", "sheep", "cow"]


# ---------------------------------------------------------------- capture

class LiveSource:
    """Keeps only the newest frame of a live stream, reconnecting on failure."""

    def __init__(self, url: str, stop: threading.Event):
        self.url, self.stop = url, stop
        self._frame, self._seq, self._lock = None, 0, threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop.is_set():
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"[capture] cannot open {redact(self.url)}; retrying in 5s")
            while cap.isOpened() and not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    print(f"[capture] lost {redact(self.url)}; reconnecting")
                    break
                with self._lock:
                    self._frame, self._seq = frame, self._seq + 1
            cap.release()
            self.stop.wait(5)

    def frames(self, fps: float):
        last = 0
        while not self.stop.is_set():
            start = time.time()
            with self._lock:
                frame, seq = self._frame, self._seq
            if frame is not None and seq != last:
                last = seq
                yield frame, start
            self.stop.wait(max(0.0, 1.0 / fps - (time.time() - start)))


def file_frames(path: str, fps: float, stop: threading.Event):
    """Every frame of a video file at the sampling rate, timestamped in video time."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video file {path}")
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 25) / fps))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    i = 0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield frame, i / src_fps
        i += 1
    cap.release()


def redact(url: str) -> str:
    p = urllib.parse.urlsplit(url)
    return url.replace(p.netloc, p.netloc.split("@")[-1]) if "@" in p.netloc else url


# ---------------------------------------------------------------- events

class EventSink:
    def __init__(self, out_dir: Path, notify_cfg: dict, retain_days: float):
        self.dir, self.notify_cfg, self.retain_days = out_dir, notify_cfg, retain_days
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = self.dir / "events.jsonl"
        self._lock = threading.Lock()
        self.live: dict[str, bytes] = {}

    def emit(self, camera: str, event, frame, wall: float):
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(wall))
        rel = f"{camera}/{stamp}-{event.label}-{event.track_id}.jpg"
        path = self.dir / rel
        path.parent.mkdir(exist_ok=True)
        shot = frame.copy()
        draw(shot, [(event.box, event.label, True)])
        cv2.imwrite(str(path), shot, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall)),
               "camera": camera, "label": event.label,
               "score": round(event.score, 2), "image": rel}
        with self._lock, self.log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[event] {rec['time']} {camera}: {event.label} ({rec['score']}) -> {rel}")
        threading.Thread(target=self._notify, args=(rec, path), daemon=True).start()

    def recent(self, n: int = 60) -> list[dict]:
        if not self.log.exists():
            return []
        lines = self.log.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(line) for line in reversed(lines)]

    def prune(self):
        if self.retain_days <= 0:
            return
        cutoff = time.time() - self.retain_days * 86400
        for p in self.dir.glob("*/*.jpg"):
            if p.stat().st_mtime < cutoff:
                p.unlink()

    def _notify(self, rec: dict, image: Path):
        title = f"{rec['label'].capitalize()} at {rec['camera']}"
        try:
            ntfy = self.notify_cfg.get("ntfy", {})
            if ntfy.get("url"):
                headers = {"Title": title, "Filename": image.name, "Tags": "camera"}
                if ntfy.get("token"):
                    headers["Authorization"] = f"Bearer {ntfy['token']}"
                urllib.request.urlopen(urllib.request.Request(
                    ntfy["url"], data=image.read_bytes(), headers=headers, method="PUT"),
                    timeout=15)
            tg = self.notify_cfg.get("telegram", {})
            if tg.get("token") and tg.get("chat_id"):
                boundary = uuid.uuid4().hex
                parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
                         .encode() for k, v in (("chat_id", tg["chat_id"]),
                                                ("caption", f"{title} — {rec['time']}"))]
                parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="photo"; '
                              f'filename="{image.name}"\r\nContent-Type: image/jpeg\r\n\r\n')
                             .encode() + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode())
                api = tg.get("api", "https://api.telegram.org")
                urllib.request.urlopen(urllib.request.Request(
                    f"{api}/bot{tg['token']}/sendPhoto", data=b"".join(parts),
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}),
                    timeout=15)
        except Exception as e:  # a failed push must never stop detection
            print(f"[notify] failed: {type(e).__name__}: {e}")


def draw(frame, tracks):
    for box, label, alerted in tracks:
        x1, y1, x2, y2 = (int(v) for v in box)
        color = (0, 0, 255) if alerted else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ---------------------------------------------------------------- dashboard

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<meta http-equiv=refresh content=10><title>mi-cam-watch</title>
<style>body{font:15px system-ui;margin:16px;background:#111;color:#eee}h2{font-size:16px;margin:20px 0 8px}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
figure{margin:0;background:#1c1c1c;border-radius:8px;overflow:hidden}img{width:100%%;display:block}
figcaption{padding:6px 10px}small{color:#999}</style>
<h2>Live</h2><div class=g>%s</div><h2>Recent events</h2><div class=g>%s</div>"""


def make_handler(sink: EventSink, cameras: list[str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            if path == "/":
                e = html.escape
                live = "".join(f'<figure><img src="/live/{e(urllib.parse.quote(c))}.jpg">'
                               f"<figcaption>{e(c)}</figcaption></figure>" for c in cameras)
                evs = "".join(
                    f'<figure><a href="/events/{e(urllib.parse.quote(r["image"]))}">'
                    f'<img loading=lazy src="/events/{e(urllib.parse.quote(r["image"]))}"></a>'
                    f'<figcaption>{e(r["label"])} · {e(r["camera"])}<br>'
                    f'<small>{e(r["time"])} · {r["score"]}</small></figcaption></figure>'
                    for r in sink.recent()) or "<small>none yet</small>"
                return self._send((PAGE % (live, evs)).encode(), "text/html; charset=utf-8")
            if path.startswith("/live/") and path.endswith(".jpg"):
                jpg = sink.live.get(path[6:-4])
                if jpg:
                    return self._send(jpg, "image/jpeg")
            if path.startswith("/events/") and path.endswith(".jpg"):
                target = (sink.dir / path[8:]).resolve()
                if target.is_relative_to(sink.dir.resolve()) and target.is_file():
                    return self._send(target.read_bytes(), "image/jpeg")
            self._send(b"not found", "text/plain", 404)

    return Handler


# ---------------------------------------------------------------- main

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def run_camera(name, url, pipeline, sink, fps, stop, rotate=0):
    is_file = "://" not in url
    frames = file_frames(url, fps, stop) if is_file else LiveSource(url, stop).frames(fps)
    for frame, t in frames:
        if ROTATIONS[rotate] is not None:
            frame = cv2.rotate(frame, ROTATIONS[rotate])
        events = pipeline.process(frame, t)
        for ev in events:
            sink.emit(name, ev, frame, time.time())
        if not is_file:
            view = frame.copy()
            draw(view, pipeline.tracks_view)
            ok, jpg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                sink.live[name] = jpg.tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(HERE / "config.toml"))
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"{cfg_path} not found — copy config.example.toml to config.toml")
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8-sig"))
    cameras = cfg.get("camera", [])
    if not cameras:
        raise SystemExit("config has no [[camera]] entries")
    names = [c["name"] for c in cameras]
    for c in cameras:
        if c.get("rotate", 0) not in ROTATIONS:
            raise SystemExit(f"camera {c['name']!r}: rotate must be 0, 90, 180 or 270")
    if len(set(names)) != len(names) or any("/" in n or n.startswith(".") for n in names):
        raise SystemExit("camera names must be unique and must not contain '/'")

    det_cfg = cfg.get("detector", {})
    detector = YoloxDetector(
        resolve_model(det_cfg.get("model", "yolox_s"), HERE / "models"),
        det_cfg.get("classes", DEFAULT_CLASSES),
        confidence=det_cfg.get("confidence", 0.45),
        threads=det_cfg.get("threads", 0))
    tuning = cfg.get("tuning", {})
    unknown = set(tuning) - {f.name for f in fields(Settings)}
    if unknown:
        raise SystemExit(f"unknown [tuning] keys: {sorted(unknown)}")
    fps = cfg.get("fps", 4)

    base = cfg_path.resolve().parent
    sink = EventSink(base / cfg.get("events_dir", "events"), cfg.get("notify", {}),
                     cfg.get("retain_days", 14))
    sink.prune()
    stop = threading.Event()
    threads = [threading.Thread(target=run_camera, daemon=True, args=(
        c["name"], c["url"], Pipeline(detector, Settings(**tuning)), sink, fps, stop,
        c.get("rotate", 0)))
        for c in cameras]
    for th in threads:
        th.start()

    web = cfg.get("web", {})
    if web.get("enabled", True):
        host, port = web.get("host", "127.0.0.1"), web.get("port", 8765)
        server = ThreadingHTTPServer((host, port), make_handler(sink, names))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"dashboard: http://{host}:{port}/")
    print(f"watching {len(cameras)} camera(s) for: {', '.join(detector.labels)}")

    try:
        next_prune = time.time() + 6 * 3600
        while any(th.is_alive() for th in threads):
            time.sleep(0.5)
            if time.time() > next_prune:
                sink.prune()
                next_prune = time.time() + 6 * 3600
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()

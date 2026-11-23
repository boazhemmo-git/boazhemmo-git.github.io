# mi-cam-watch

Watches Xiaomi "Mi Home" cameras and alerts when a **person or animal moves**,
while ignoring trees, shadows, rain and anything else the wind pushes around.

```
Mi Home camera ──(Xiaomi P2P)──> go2rtc ──(RTSP)──> watch.py ──> snapshot + push + dashboard
```

It runs on a machine on the same network as the cameras (PC, mini-PC, Raspberry
Pi 4/5). It cannot run on GitHub Pages or any other static host.

## Why it ignores wind

1. **Motion gate** – background subtraction; if nothing changes, nothing else runs.
2. **Object detector** – YOLOX only outputs people/animals. A swaying tree is
   motion but not an object of interest, so it can never alert on its own.
3. **Track confirmation** – a detection must persist for several frames, have
   changing pixels inside its box, *and* change position. This removes
   one-frame misfires and statues or sleeping pets in front of a waving bush.

## Setup

1. **Bridge the camera with [go2rtc](https://github.com/AlexxIT/go2rtc) ≥ 1.9.14.**
   Mi Home has no public video API; go2rtc's `xiaomi` source logs in to your
   Xiaomi account and re-publishes the camera as RTSP. Start it, open
   `http://localhost:1984`, choose *Add → Xiaomi*, log in, pick your cameras.
   See `go2rtc.example.yaml`. Not every model works — check the
   [compatibility list](https://github.com/AlexxIT/go2rtc/issues/1982) first.
   Confirm you can see video in the go2rtc web UI before continuing.
2. **Install** (Python 3.11+):
   ```
   pip install -r requirements.txt
   cp config.example.toml config.toml     # then edit camera names/URLs
   ```
3. **Run:** `python watch.py` (on Windows, double-click `start.bat` instead; it also
   silences FFmpeg's decoder warnings) — dashboard at <http://127.0.0.1:8765/>.
   The model (~35 MB) is downloaded and checksum-verified on first start.

Snapshots land in `events/<camera>/`, with a log in `events/events.jsonl`.

## Run at Windows start-up

First restrict go2rtc to this computer: its web UI has no password and can
read your Xiaomi token and edit its config. Put this at the top of `go2rtc.yaml`:

```yaml
api:
  listen: "127.0.0.1:1984"
rtsp:
  listen: "127.0.0.1:8554"
webrtc:
  listen: "127.0.0.1:8555"
```

Then, in a Command Prompt opened with *Run as administrator*:

```
python autostart.py install --go2rtc C:\go2rtc\go2rtc_win64\go2rtc.exe
python autostart.py start
```

Both programs then start at every boot, before anyone logs in, restart if they
exit, and write to `logs\watch.log` and `logs\go2rtc.log`. Don't also run
`start.bat` or `go2rtc.exe` by hand while the task is running.
`python autostart.py stop` / `status` / `remove` manage it.

For a laptop: set it to never sleep when plugged in and to do nothing when the
lid closes, and enable a battery charge limit if the manufacturer offers one.

## Testing without a camera

A camera `url` may be a video file path; the file is processed start to finish
and the program exits. `python -m unittest test_pipeline` runs the logic tests.

## Tuning

| Symptom | Change |
|---|---|
| Alerts on things that aren't alive | raise `detector.confidence` (0.55–0.6) |
| Misses small/distant subjects | lower `confidence`, raise camera `subtype` to HD in go2rtc |
| CPU too high | lower `fps`, or `model = "yolox_tiny"` |
| Too many repeat alerts | raise `tuning.cooldown_s` |

## Known limits

- Someone who is already in view and motionless when the program starts is not
  reported until they move.
- Night/IR footage is detected less reliably than daylight.
- The dashboard has no authentication; keep `host = "127.0.0.1"` or put it
  behind something that does.
- `config.toml`, `go2rtc.yaml` (holds your Xiaomi token) and `events/` are
  git-ignored. Keep it that way — this repository is public.

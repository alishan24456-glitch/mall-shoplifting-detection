# Mall Shoplifting-Detection — Behaviour-Risk Flagging

A retail loss-prevention assistant that watches a store / mall camera feed and
flags **behaviour risk** for staff review. It tracks every person with
YOLOv8 + ByteTrack, watches the items they carry, and raises a timestamped
flag when a person shows one or more heuristic risk signals.

> **Honest framing:** this system flags behaviour *risk* — it does **not**
> and cannot prove theft. Innocent shoppers dwell at shelves, reach for
> products, put bags down and hurry out of stores every day. Every flag is
> a prompt for a human to review the footage, nothing more.

## The four risk signals

Each signal adds points to a per-person risk score. Boxes are coloured by
level: **green** = LOW (< 30), **amber** = WATCH (30–59), **red** = RISK
FLAG (≥ 60, alert banner + event-log entry).

| Signal | What the pipeline actually measures | Points |
|---|---|---|
| **LONG DWELL** | The person's centroid stays inside one floor zone (frame split into a 6×4 grid) longer than the dwell limit (default 10 s) | +25, +1/s beyond the limit (cap +15) |
| **REPEATED REACH** | While the person is nearly stationary in one zone, their box repeatedly grows ≥12% taller than its recent baseline and settles back — the box-geometry signature of arm raises toward a shelf; fires at the 3rd reach in the same zone stay | +20, +5 per extra reach (cap +20) |
| **ITEM VANISHED** | A carried item (COCO classes: backpack, umbrella, handbag, suitcase, sports ball, bottle, cup) is stably associated with the person for ≥ 1 s, then no item is associated for ≥ 2 s while the person is still tracked — a concealment-style signal | +35 |
| **RAPID EXIT** | A person already at WATCH level or above disappears from the scene with a recent centroid speed ≥ 260 px/s | +20 |

Every signal and every RISK FLAG is printed to the console and appended to
a timestamped event log (`events.log`), and the run ends with a summary:
frames processed, people tracked, flags by type, longest dwell, highest
score.

## Run it (Windows CMD)

```
pip install -r requirements.txt
python mall_shoplifting_detection.py                  REM opens a video file chooser
python mall_shoplifting_detection.py --source pick    REM same chooser, explicitly
python mall_shoplifting_detection.py --source mall.mp4
python mall_shoplifting_detection.py --source 0       REM webcam
python mall_shoplifting_detection.py --simulate       REM rendered store scene
```

Press **Q** to quit. The annotated video is written to `output.mp4`
(change with `--output`); the event log goes to `events.log`
(change with `--log`). Useful options: `--dwell 15` (dwell limit in
seconds), `--conf 0.30`, `--weights yolov8n.pt` (downloaded automatically
by Ultralytics on first use if missing), `--max-seconds 30`,
`--no-display` for headless runs.

## Modes

- **Real mode** — YOLOv8 detects people and carried items; ByteTrack
  (Ultralytics) supplies person track IDs.
- **Simulation mode** (`--simulate`) — a store scene is rendered with
  OpenCV: shelves, four shoppers (a browser, a shelf-dweller who reaches
  repeatedly, a stander, and a shopper whose handbag disappears mid-stay
  before a hurried exit). The scene publishes person/item boxes derived
  from the rendered figures through a small centroid tracker, producing
  exactly the same detection stream as real mode. The risk engine only
  ever sees that stream — it never peeks at the scene's scripted ground
  truth.

## Verified demo runs

**Simulation (30 s, 900 frames):** 4 people tracked, **2 RISK FLAGs**;
LONG DWELL ×3, REPEATED REACH ×1, ITEM VANISHED ×1, RAPID EXIT ×1
(carrier left at 317 px/s); longest dwell 17.1 s; highest score 80.

**Real footage (busy mall atrium, 720p, first 30 s / 915 frames):**
**Real footage (busy mall atrium, 720p, first 30 s / 916 frames, default
ByteTrack):** 216 person tracks — ByteTrack fragments IDs heavily in a
crowd this dense — with **LONG DWELL ×1** (ID 278, zone D2, flagged at
27.5 s), longest dwell 11.3 s, highest score 26, and no RISK FLAG.
REPEATED REACH, ITEM VANISHED and RAPID EXIT did **not** fire on this
clip, and that is reported as-is: at this wide angle YOLOv8n rarely
resolves carried items on people this small, and stationary shoppers'
box-height changes stay under the reach threshold. A crowd-tuned tracker
config (`bytetrack_mall.yaml`, longer track buffer, selectable with
`--tracker`) was also tried: it cut fragmentation to 53 tracks, but the
followed boxes wandered between neighbours, zone stays shortened
(max dwell 5.5 s) and no dwell flag fired — so the default-tracker run
above is the shipped demo. The simulation run is where the full
four-signal chain is demonstrated end to end.

## Files

- `mall_shoplifting_detection.py` — the whole system (single file)
- `bytetrack_mall.yaml` — optional crowd-tuned ByteTrack config
- `requirements.txt`, `.gitignore`

Author: Abdul Hanan — Computer Vision & AI Developer

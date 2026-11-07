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

**Real footage — primary demo (single shopper in a store aisle,
1280×676, full clip 20.1 s / 503 frames, default settings):** one man
pushing a shopping cart between stocked shelves, tracked as a single
stable ID for essentially the whole clip — **2 person tracks** in total,
versus 216 in the crowd test below. Results: **LONG DWELL ×1** (ID 1,
zone C4, flagged at 16.0 s), longest dwell 14.0 s, highest score 29,
and no RISK FLAG — the shopper ended just below the WATCH threshold of
30. REPEATED REACH, ITEM VANISHED and RAPID EXIT did **not** fire, and
that is reported as-is: he carries no detectable item (the cart is not
a COCO carried class) and never exits at speed. What the sparse scene
does show cleanly is the tracking quality the risk engine depends on —
one unbroken track, a steadily counting dwell timer and a per-zone stay
measured end to end. This run ships as
`mall-shoplifting-detection-demo.mp4`.

**Real footage — dense-crowd test (busy mall atrium, 720p, first 30 s /
916 frames, default ByteTrack):** 216 person tracks — ByteTrack
fragments IDs heavily in a crowd this dense — with **LONG DWELL ×1**
(ID 278, zone D2, flagged at 27.5 s), longest dwell 11.3 s, highest
score 26, and no RISK FLAG. REPEATED REACH, ITEM VANISHED and RAPID
EXIT did **not** fire on this clip either: at this wide angle YOLOv8n
rarely resolves carried items on people this small, and stationary
shoppers' box-height changes stay under the reach threshold. A
crowd-tuned tracker config (`bytetrack_mall.yaml`, longer track buffer,
selectable with `--tracker`) was also tried: it cut fragmentation to
53 tracks, but the followed boxes wandered between neighbours, zone
stays shortened (max dwell 5.5 s) and no dwell flag fired. This run is
kept as `mall-shoplifting-detection-demo-crowded.mp4`. The simulation
run is where the full four-signal chain is demonstrated end to end.

**Real footage — small-group segment (same mall clip, seconds
9.0–29.0, full frame 1280×720, 20.0 s / 611 frames, default settings;
segment saved as `real-footage/retail-smallshop-segment.mp4`):** the
window was picked by scanning every frame of the mall clip with the
same detector — in this window the engine typically sees a small
group of ground-floor shoppers (median 7 people per frame, mean 7.4,
three quarters of frames between 6 and 9), between the 1-person
warehouse demo above and the full-atrium crowd. No spatial crop was
used: crops were measured during selection and rejected, because
enlarging part of this high-angle frame makes *more* shoppers
detectable, not fewer (an 854×480 crop of the foreground measured a
median of 17 detections per frame). Results on the segment: 149
person tracks — fragmentation is still heavy, because the tracked
shoppers constantly occlude each other — with **LONG DWELL ×1**
(ID 150, zone D2: a man standing at the glass railing, flagged at
17.9 s of the segment), longest dwell 12.0 s, highest score 26, and
no RISK FLAG. REPEATED REACH, ITEM VANISHED and RAPID EXIT did
**not** fire on this segment either, and that is reported as-is —
the dwell is the one behaviour this scene cleanly demonstrates.
This run ships as `mall-shoplifting-detection-demo-smallshop.mp4`
with its log `mall-shoplifting-detection-events-smallshop.log`.

## Files

- `mall_shoplifting_detection.py` — the whole system (single file)
- `bytetrack_mall.yaml` — optional crowd-tuned ByteTrack config
- `requirements.txt`, `.gitignore`

Author: Abdul Hanan — Computer Vision & AI Developer

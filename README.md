# Smart Room

A Raspberry Pi project that watches a room, recognizes who walks in, and reacts:
it starts that person's Spotify playlist and sends me a Telegram message (a photo
if it doesn't recognize the face). Once a known person is in frame, they can also
control playback with hand gestures - recognized by a classifier I trained on top
of the camera's on-sensor pose estimation.

The point of the project was to actually use the Raspberry Pi AI Camera (IMX500)
for on-device inference instead of treating it as a webcam, and to do some real ML
on top of it rather than just calling a pretrained library.

README ASSISTED BY CLAUDE FOR IMPROVED READABILITY

## What it does

- Recognizes faces and greets known people by name
- Plays a per-person Spotify playlist on a chosen device
- Telegram alerts on entry (with a photo when the face is unknown)
- Hand-gesture control for recognized people, via a trained model:
  - left hand up -> next track
  - right hand up -> play / pause
  - both hands up -> that person's favorite song
  - t-pose -> that person's love song
- Handles more than one person in frame (greets everyone, music for the first)

## Hardware

- Raspberry Pi 5 (8GB)
- Raspberry Pi AI Camera (Sony IMX500 sensor, neural processing on the sensor itself)
- A Spotify Connect target 

## How it works

Two things run off the one camera at the same time:

- **Face recognition** on the Pi's CPU (`face_recognition` / dlib). The encodings
  for the training photos are computed once and cached to disk, so startup is fast
  after the first run.
- **Pose estimation** on the camera's NPU using a pretrained HigherHRNet model. The
  Pi never runs that model - the sensor runs it and returns the body keypoints in
  the frame metadata.

The main loop grabs one frame, reads the pose keypoints from its metadata every
cycle, and runs the slower face recognition about once a second. Gestures only do
anything while a known person is recognized.

## The gesture model (the ML part)

The pose model is pretrained - it just gives 17 body keypoints. The actual work is
everything built on top of those points.

### Features
Each pose becomes 14 numbers: the (x, y) of 7 upper-body joints (nose, shoulders,
elbows, wrists). Before they're used, every pose is **normalized** - recentered on
the midpoint between the shoulders and scaled by shoulder width - so the same
gesture produces the same numbers no matter where you stand or how far from the
camera you are. Without that, the model would just memorize pixel positions and
break the moment you moved.

### Model
A scikit-learn RandomForest (200 trees). For ~1700 keypoint vectors it's the right
fit: fast to train, runs instantly on CPU, needs no feature scaling, and is robust
on small tabular data. Five classes: neutral, left_hand, right_hand, both_hands,
t_pose.

### How it was validated (and why I don't report 100%)
Same-session cross-validation hit ~99%, but that's optimistic - held-out frames
from the same recording look almost identical to the training frames. The honest
test is **cross-session**: train on one recording, test on a separate one collected
in different conditions (position, distance, clothing). That gave **93%**, which is
the number I trust.

I also iterated, and the iteration is the interesting part:

| version | data | cross-session | issue |
|---|---|---|---|
| v1 | 1500 balanced | 93% | right_hand over-predicted (precision 0.75) |
| v2 | +200 neutral | 89% | over-corrected - neutral started eating left_hand / t_pose |
| v3 | v2 + class_weight | 88% | barely changed -> the problem is feature overlap, not count imbalance |

The takeaway: poses like "hand near face" genuinely overlap with real gestures in
feature space, so labeling them neutral just creates contradictory training
examples that no model can separate. Ambiguity is better handled at **runtime** (a
confidence gate that ignores unsure predictions) than forced into the labels.

### What the model actually learned
Feature importance shows the **wrist and elbow coordinates dominate**, while the
shoulders are near zero - which makes sense, because the shoulders are the
normalization anchor (roughly fixed after centering), so the gesture information
lives in the hands. Reading one tree confirms it learned real body logic: "is the
left elbow up? -> is the right elbow also up? -> both_hands vs left_hand," and so
on. Visuals are in `docs/` (feature importances, confusion matrix, an example tree).

### Runtime safety
Live, a gesture only fires if the model is confident enough
(`GESTURE_CONFIDENCE_THRESHOLD`) and isn't within a short cooldown of the last
command - so an ambiguous pose does nothing instead of misfiring.

## Project layout

```
smart_room_project/
├── main.py                 entry point: camera loop, ties everything together
├── telegram_control.py     telegram start (runs as a service)
├── requirements.txt
├── people.py.example       copy to people.py and fill in your own data
├── deply/                  systemmd unit file
│   ├──
├── smart_room/
│   ├── recognizer.py       load/cache faces, identify who's in frame
│   ├── gestures.py         normalize keypoints -> features, classify, map to actions
│   ├── action.py           what happens on a detection (spotify + telegram)
│   ├── spotify_control.py  playback control
│   ├── telegram_notify.py  text + photo alerts
│   └── config.py           paths + tuning constants (pulls personal data from people.py)
├── tools/
│   ├── capture_faces.py    grab face training photos
│   ├── collect_gestures.py record labeled pose samples
│   ├── train_gestures.py   train + save the gesture model
│   ├── eval_gestures.py    cross-session evaluation
│   └── visualize_model.py  generate the model visuals
├── models/                 trained gesture model (encodings cache is gitignored)
├── data/                   pose datasets (gitignored)
├── docs/                   model visuals
├── known_faces/            face training photos (gitignored)
└── captures/               snapshots of unknown faces (gitignored)
```

## Setup

System packages first, these aren't pip packages:

```
sudo apt update
sudo apt install -y python3-picamera2 imx500-all
sudo reboot      # needed so the imx500 loads its firmware
```

Then the Python side:

```
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt    # dlib builds from source, give it a few minutes
```

Secrets go in a `.env` file in the project root (not committed):

```
SPOTIFY_CLIENT_ID=...
SPOTIFY_CLIENT_SECRET=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Your people, playlists, and per-person gesture songs go in `people.py` (copy the
template and fill it in):

```
cp people.py.example people.py
```

Capture face training photos for each person (run from the project root):

```
python3 tools/capture_faces.py
```

## Running

```
source venv/bin/activate
python3 main.py
```

First run uploads the pose model to the camera and builds the face cache. Ctrl+C
to stop. Recognized person -> their playlist + a Telegram message; then their
gestures control playback.

## Running on boot (phone control)

Rather than auto-running the room on boot, a small controller (`telegram_control.py`)
runs as a systemd service and lets me start/stop the room from my phone over
Telegram - so the camera only runs when I ask for it.

Commands (sent to the bot from my own chat):

- `/start` - launch the room
- `/stop` - shut it down (releases the camera cleanly)
- `/status` - is it running?

The controller is owner-only - it ignores any chat that isn't my `TELEGRAM_CHAT_ID`,
so a random person who finds the bot can't turn the camera on. It launches `main.py`
as a subprocess (logging to `room.log`) and stops it with SIGINT so the camera tears
down properly.

room.log has been gitignored due to sensitive information

Set it up as a service so it starts on boot (adjust the user/paths for your setup):

sudo systemctl daemon-reload
sudo systemctl enable --now smart-room.service

Logs: `journalctl -u smart-room.service -f` for the controller, `tail -f room.log`
for the room itself. (A copy of the unit file is in `deploy/` for reference.)

## Retraining the gesture model

The repo ships with a trained `models/gesture_model.joblib`, but to train your own:

```
# 1. collect labeled samples (hold each pose, vary position/distance)
python3 -m tools.collect_gestures            # -> data/gesture_data.csv

# 2. collect a separate session for honest evaluation
python3 -m tools.collect_gestures gesture_test.csv

# 3. train (prints accuracy, saves the model)
python3 tools/train_gestures.py

# 4. cross-session evaluation
python3 tools/eval_gestures.py

# 5. (optional) regenerate the visuals
python3 tools/visualize_model.py
```

## Notes and known limitations

- `right_hand` is slightly over-predicted - kept on purpose as a documented
  precision/recall tradeoff (see the table above).
- Raising both hands can briefly fire a single-hand gesture first, since one hand
  crosses the threshold before the other.
- Pose keypoints need short sleeves / fitted clothing - long, loose sleeves drop
  the shoulder keypoint confidence and the pose can't be read.
- Volume gestures were dropped: many devices (Apple TV, Bluetooth speakers) return
  `VOLUME_CONTROL_DISALLOW` and won't accept remote volume over Spotify.
- Side-profile faces recognize poorly (a limitation of the dlib face model).
- Only the first recognized person's music plays, since one device plays one thing.

## Notes on the dependencies

`numpy` is pinned to 1.24.2 on purpose - the system camera stack (`picamera2`,
`simplejpeg`) is built against it, and a newer numpy breaks it. `scikit-learn` is
pinned to a 1.x release for the same reason (newer ones pull numpy 2). OpenCV is
the headless build to avoid a Qt plugin conflict.
```

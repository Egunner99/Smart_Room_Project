# Smart Room

A Raspberry Pi project that watches a room, recognizes who walks in, and reacts:
it starts that person's Spotify playlist and sends me a Telegram message. If it
sees someone it doesn't know, it sends a photo instead. Once a known person is in
frame they can also control playback with hand gestures.

The point of the project was to actually use the Raspberry Pi AI Camera (IMX500)
for on-sensor inference, not just treat it as a webcam.

## What it does

- Detects and recognizes faces, and greets known people by name
- Plays a per-person Spotify playlist on a chosen device
- Telegram alerts on entry (with a photo when the face is unknown)
- Hand gesture control once a known person is present:
  - right hand up: next track
  - left hand up: play / pause
- Handles more than one person in frame (greets everyone, plays music for the first)

## Hardware

- Raspberry Pi 5 (8GB)
- Raspberry Pi AI Camera (IMX500 sensor, has its own neural processing chip)
- A Spotify Connect target (mine is an Apple TV, set by device type in config)

## How it works

There are two things running off the one camera:

- **Face recognition** runs on the Pi's CPU using the `face_recognition` (dlib)
  library. Encodings for the training photos are computed once and cached to disk
  (`encodings.npz`), so startup is fast after the first run.
- **Pose estimation** runs on the camera's NPU using a pretrained HigherHRNet
  model. The Pi never runs that model - the sensor does it and hands back the
  keypoints in the frame metadata. The gesture logic on top of those keypoints
  (which wrist is above which shoulder, with confidence checks and a bit of
  debouncing) is the custom part.

The main loop grabs one frame, reads the pose keypoints from its metadata every
cycle, and runs the slower face recognition about once a second. Gestures only
do anything while a known person is recognized.

## Project layout

```
smart_room_project/
├── main.py                 entry point, the camera loop
├── requirements.txt
├── people.py.example       copy to people.py and fill in your own data
├── smart_room/
│   ├── recognizer.py       load/cache faces, identify who's in frame
│   ├── gestures.py         turn pose keypoints into music commands
│   ├── action.py           what happens on a detection (spotify + telegram)
│   ├── spotify_control.py  playback control
│   ├── telegram_notify.py  text + photo alerts
│   └── config.py           constants, pulls playlists from people.py
├── tools/
│   └── capture_faces.py    grab training photos
├── tests/                  scratch scripts from building it
├── known_faces/            training photos (gitignored)
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

Your people and their playlists go in `people.py`:

```
cp people.py.example people.py
# then edit people.py with names and Spotify URIs
```

Capture some training photos for each person (run from the project root):

```
python3 tools/capture_faces.py
```

## Running

```
source venv/bin/activate
python3 main.py
```

First run uploads the pose model to the camera (takes a bit) and builds the face
cache. Ctrl+C to stop.

## Notes and things left to do

- Side profiles are weak. dlib's face model is built for front-facing faces, so
  turning your head sideways usually reads as unknown.
- Only the first recognized person's music plays, since one device can only play
  one thing. Everyone known still gets a greeting.
- No auto-start on boot yet (planned: a systemd service).
- Gestures need the hand fully lowered between repeats, since the detection
  resets on the "hand down" state.

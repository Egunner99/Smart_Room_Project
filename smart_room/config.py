#add
# orginally had personal playlist and names, made genric for public use

import os 
from people import PLAYLISTS, UNKNOWN_TRACK, FAVORITE_TRACKS, LOVE_TRACKS

#----paths----
KNOWN_FACES_DIR = "known_faces"
CAPTURES_DIR = "captures"  # directory to save captured images of unknown faces
MODELS_DIR = "models"  # directory to save trained models
POSE_MODEL = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
ENCODINGS_CACHE = "models/encodings.npz"   # cached face encodings (gitignored)

#----timing----
COOLDOWN_SECONDS = 15 # cooldown trigger for the same person counts
FACE_INTERVAL = 1.0           # seconds between face-recognition passes
GESTURE_COOLDOWN = 3.0 # be able to move positions

# ---playback---
PLAYBACK_DEVICE = "Smartphone"  # use spotify control to view devicetypes active

#---gesture parameters---
POSE_THRESHOLD = 0.5           # min confidence to count a pose detection as a person
GESTURE_MIN_CONFIDENCE = 0.5   # min confidence for one joint to be trusted
GESTURE_RAISE_MARGIN = 30      # pixels a wrist must be above its shoulder to count
GESTURE_MODEL = "models/gesture_model.joblib"  # trained gesture model
GESTURE_CONFIDENCE_THRESHOLD = 0.6  # min confidence for gesture prediction to be trusted

# ---face recognition tuning---
RECOGNITION_TOLERANCE = 0.5   # face match strictness, lower = stricter


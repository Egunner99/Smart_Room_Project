#working
import numpy as np
import joblib
from .spotify_control import next_track, toggle_play_pause, play_track
from .config import PLAYBACK_DEVICE, GESTURE_MIN_CONFIDENCE, GESTURE_RAISE_MARGIN, GESTURE_MODEL, GESTURE_CONFIDENCE_THRESHOLD, FAVORITE_TRACKS, LOVE_TRACKS

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10

_model = None

def classify_gesture(person):
    #return gesture label
    global _model
    if _model is None:
        _model = joblib.load(GESTURE_MODEL)
    features = extract_features(person)
    if features is None:
        return None
    prediction = _model.predict_proba([features])[0] #floats
    i = prediction.argmax()
    if prediction[i] < GESTURE_CONFIDENCE_THRESHOLD:
        return None
    return _model.classes_[i]


# def hand_raised(person):
#     ls, lw = person[L_SHOULDER], person[L_WRIST]
#     if ls[2]  > GESTURE_MIN_CONFIDENCE and lw[2] > GESTURE_MIN_CONFIDENCE and lw[1] < ls[1] - GESTURE_RAISE_MARGIN:
#         return "left hand raised"
#     rs, rw = person[R_SHOULDER], person[R_WRIST]
#     if rs[2] > GESTURE_MIN_CONFIDENCE and rw[2] > GESTURE_MIN_CONFIDENCE and rw[1] < rs[1] - GESTURE_RAISE_MARGIN:
#         return "right hand raised" 
    
#     return None

def handle_gesture(gesture, person):
    if gesture == "right_hand":
        toggle_play_pause(PLAYBACK_DEVICE)
    elif gesture == "left_hand":
        next_track(PLAYBACK_DEVICE)
    elif gesture == "both_hands":
        track = FAVORITE_TRACKS.get(person)
        if track:
            play_track(track, PLAYBACK_DEVICE)
    elif gesture == "t_pose":
        track = LOVE_TRACKS.get(person)
        if track:
            play_track(track, PLAYBACK_DEVICE)



FEATURE_KEYPOINTS = [0,5,6,7,8,9,10] 
FEATURE_MIN_CONFIDENCE = 0.2

def extract_features(person):
    ls, rs = person[L_SHOULDER], person[R_SHOULDER] #this is the anchor, shoulders both need to be visible if not 
    if ls[2] < FEATURE_MIN_CONFIDENCE or rs[2] < FEATURE_MIN_CONFIDENCE:
        return None

    center = (ls[:2] + rs[:2]) / 2 # midpoint between shoulders, only take x and y coordinates
    scale = np.linalg.norm(rs[:2] - ls[:2]) #added gauge to scale features 
    #added catch case for scale too small = too far away or not visible
    if scale < 1e-6: 
        return None

    features = [] 
    for i in FEATURE_KEYPOINTS:
        kp = person[i]
        if kp[2] < FEATURE_MIN_CONFIDENCE: # kp[2] is confidence score of NPU, if less NONE
            return None
        else:
            norm = (kp[:2] - center) / scale #scaling again, solves distance and position 
            features.extend([float(norm[0]), float(norm[1])])
    return features

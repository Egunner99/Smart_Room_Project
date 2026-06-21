#working

from .spotify_control import next_track, toggle_play_pause
from .config import PLAYBACK_DEVICE

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10
MIN_CONFIDENCE = 0.5
RAISE_MARGIN = 30 #strict

def hand_raised(person):
    ls, lw = person[L_SHOULDER], person[L_WRIST]
    if ls[2]  > MIN_CONFIDENCE and lw[2] > MIN_CONFIDENCE and lw[1] < ls[1] - RAISE_MARGIN:
        return "left hand raised"
    rs, rw = person[R_SHOULDER], person[R_WRIST]
    if rs[2] > MIN_CONFIDENCE and rw[2] > MIN_CONFIDENCE and rw[1] < rs[1] - RAISE_MARGIN:
        return "right hand raised" 
    
    return None

def handle_gesture(gesture):
    if gesture == "left hand raised":
        toggle_play_pause(PLAYBACK_DEVICE)
    elif gesture == "right hand raised":
        next_track(PLAYBACK_DEVICE)
# working 

import os
import cv2

from .spotify_control import play_for_person
from .telegram_notify import send_message, send_photo
from .config import PLAYLISTS, UNKNOWN_TRACK, CAPTURES_DIR, PLAYBACK_DEVICE

def handle_person(name, frame, play_music=True):
    if name.lower() == "unknown":
        capture_path = os.path.join(CAPTURES_DIR, "unknown.jpg")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # UPDATE color conversion for saving
        cv2.imwrite(capture_path, bgr)  # save the captured image
        if play_music:
            play_for_person("Unknown", UNKNOWN_TRACK, device_type=PLAYBACK_DEVICE)
        send_message("Unknown person detected")
        send_photo(capture_path, caption="Unknown person detected in the Smart Room")


    else:
        if play_music:
            playlist = PLAYLISTS.get(name, UNKNOWN_TRACK) #default to unknown
            if playlist:
                play_for_person(name, playlist, device_type=PLAYBACK_DEVICE)
        send_message(f"{name} detected in the Smart Room")

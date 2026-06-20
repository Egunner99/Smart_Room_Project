# working 

import os
import cv2

from .spotify_control import play_for_person
from .telegram_notify import send_message, send_photo
from .config import PLAYLISTS, UNKNOWN_TRACK, CAPTURES_DIR

def handle_person(name, frame):
    if name.lower() == "unknown":
        capture_path = os.path.join(CAPTURES_DIR, "unknown.jpg")
        cv2.imwrite(capture_path, frame)  # save the captured image
        play_for_person("Unknown", UNKNOWN_TRACK, device_type="TV")
        send_message("Unknown person detected in the smart room.")
        send_photo(capture_path, caption="Captured image of the unknown person.")

    else:
        playlist = PLAYLISTS.get(name)
        if playlist:
            play_for_person(name, playlist, device_type="TV")
        send_message(f"{name} walked in")

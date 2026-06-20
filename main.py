# imports for external libraries
from picamera2 import Picamera2
import face_recognition
import cv2
import time
import os

# imports from other files in project
from spotify_control import play_for_person
from telegram_notify import send_message, send_photo
from config import KNOWN_FACES_DIR, COOLDOWN_SECONDS, PLAYLISTS, UNKNOWN_TRACK

known_encodings = []
known_names = []

print("have to load known faces. takes time depending on the number of faces")

print(f"loaded {len(known_encodings)} encodings for {len(set(known_names))} people")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()
time.sleep(1)

print("running... ctrl+c to stop")

last_seen = None
last_trigger_times = {}  # cooldown tracked per person, so dad walking in right after me still triggers


def handle_person(name, frame):
    last_trigger_times[name] = time.time()

    if name == "unknown":
        cv2.imwrite("unknown_capture.jpg", frame)
        play_for_person("unknown", UNKNOWN_TRACK, device_type="TV")
        send_message("unknown person detected in the room")
        send_photo("unknown_capture.jpg", caption="unknown face")
    else:
        playlist = PLAYLISTS.get(name)
        if playlist:
            play_for_person(name, playlist, device_type="TV")
        send_message(f"{name} just walked in")


try:
    while True:
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        current_name = None

        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            distances = face_recognition.face_distance(known_encodings, encoding)

            if len(distances) > 0:
                best_match = distances.argmin()
                if matches[best_match]:
                    current_name = known_names[best_match]

        if face_encodings and current_name is None:
            current_name = "unknown"

        if current_name and current_name != last_seen:
            last_time = last_trigger_times.get(current_name, 0)
            if time.time() - last_time > COOLDOWN_SECONDS:
                print(f"recognized: {current_name}" if current_name != "unknown" else "unknown face detected")
                handle_person(current_name, frame)
            last_seen = current_name
        elif not face_encodings:
            last_seen = None

        time.sleep(0.5)

except KeyboardInterrupt:
    print("stopping")

finally:
    picam2.stop()

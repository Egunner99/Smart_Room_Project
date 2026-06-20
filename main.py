#revamped

import os
import time
import cv2

from picamera2 import Picamera2

# load from other files UPDATE NEEDED POST ORG
from smart_room.recognizer import load_known_faces, identify_in_frame
from smart_room.action import handle_person
from smart_room.config import KNOWN_FACES_DIR, COOLDOWN_SECONDS, CAPTURES_DIR

os.makedirs(CAPTURES_DIR, exist_ok=True)  # ensure captures directory exists

print("need to load known faces. takes times") # this is a bottleneck on bootup
known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
print("known faces loaded")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()
time.sleep(2)  # allow camera to warm up

print("the smart room is active. to cancel press ctrl+c or terminate the program")

last_seen = None
last_trigger_time = {}
"""
trigger time exist in a dictionary to differentiate between people
"""

try:
    while True:
        frame = picam2.capture_array()
        name = identify_in_frame(frame, known_encodings, known_names)

        if name and name != last_seen:
            last_time = last_trigger_time.get(name, 0)
            if time.time() - last_time > COOLDOWN_SECONDS:
                print(f"Identified: {name}" if name != "Unknown" else "Identified: Unknown face") # built in if to handle both
                last_trigger_time[name] = time.time()
                handle_person(name, frame)
            last_seen = name
        elif name is None:
            last_seen = None  # reset
        
        time.sleep(1)  # small delay to not overwhelm the CPU

except KeyboardInterrupt:
    print("shut down the smart room")

finally:
    picam2.stop()


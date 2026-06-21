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

last_trigger_time = {}
present = set()  # to track who is currently in the room

# trigger time exist in a dictionary to differentiate between people


try:
    while True:
        frame = picam2.capture_array()
        known_here, saw_unknown = identify_in_frame(frame, known_encodings, known_names)

        names_here = set(known_here)

        if saw_unknown:
            names_here.add("Unknown")

        #trigger music for first perosn not all
        music_person = known_here[0] if known_here else "Unknown"  
        
        for person in names_here:
            if person in present:
                continue  
            if time.time() - last_trigger_time.get(person, 0) <= COOLDOWN_SECONDS:
                continue #cooldown

            last_trigger_time[person] = time.time()  

            if person == "unknown":
                print("unknown person detected")
                handle_person("Unknown", frame, play_music=(music_person == None))
            else:
                print(f"{person} detected")
                handle_person(person, frame, play_music=(music_person == person))
        present = names_here # remember

except KeyboardInterrupt:
    print("shut down the smart room")

finally:
    picam2.stop()


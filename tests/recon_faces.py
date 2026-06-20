from picamera2 import Picamera2
import face_recognition
import cv2
import time
import os

KNOWN_FACES_DIR = "known_faces"

known_encodings = []
known_names = []

# load every photo in known_faces/ and turn it into a face encoding
# folder name = person's name, so known_faces/gio/whatever.jpg -> "gio"
print("loading known faces...")

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)

        # some photos might not have a clean face in them, skip those
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
        else:
            print(f"no face found in {filepath}, skipping")

print(f"loaded {len(known_encodings)} encodings for {len(set(known_names))} people")

# fire up the camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()
time.sleep(1)  # give it a sec to actually start before we grab frames

print("running... ctrl+c to stop")

last_seen = None  # tracks who we last saw so we don't spam the same name every loop

try:
    while True:
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        current_name = None

        for encoding in face_encodings:
            # compare this face against everyone we know
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            distances = face_recognition.face_distance(known_encodings, encoding)

            if len(distances) > 0:
                # closest match = lowest distance, this is just "who looks most like this"
                best_match = distances.argmin()
                if matches[best_match]:
                    current_name = known_names[best_match]

        # only print when something actually changes, otherwise this spams the terminal
        if face_encodings and current_name is None:
            current_name = "unknown"

        if current_name and current_name != last_seen:
            if current_name == "unknown":
                print("unknown face detected")
            else:
                print(f"recognized: {current_name}")
            last_seen = current_name
        elif not face_encodings:
            last_seen = None  # nobody in frame, reset so next person triggers a fresh print

        time.sleep(0.5)  # don't hammer the cpu, half a second between checks is plenty

except KeyboardInterrupt:
    print("stopping")

finally:
    picam2.stop()

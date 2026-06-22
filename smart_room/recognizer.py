#making
import os
import numpy as np
import cv2
import face_recognition  # this relies on the dlib library, CPU intensive
from .config import ENCODINGS_CACHE, RECOGNITION_TOLERANCE
# later to be revamped using the AI Camera 

#handling bootleneck of loading known faces
def build_encoding(known_faces_dir):
    encodings = []
    names = [] 

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(path)
            face_encodings = face_recognition.face_encodings(image)

            if not face_encodings:
                print(f"Warning: No faces found in {path}")
                continue

            encodings.append(face_encodings[0])
            names.append(person_name)

    return encodings, names

def _cache_is_fresh(known_faces_dir):
    if not os.path.exists(ENCODINGS_CACHE):
        return False

    cache_mtime = os.path.getmtime(ENCODINGS_CACHE)

    for root, _, files in os.walk(known_faces_dir):
        for filename in files:
            if os.path.getmtime(os.path.join(root, filename)) > cache_mtime:
                return False
    return True   

def load_known_faces(known_faces_dir):
    if _cache_is_fresh(known_faces_dir):
        data = np.load(ENCODINGS_CACHE)
        encoding = list(data["encodings"])
        names = [str(n) for n in data["names"]]
        print("loaded encodings from cache")
        return encoding, names

    print("building encodings from images")
    encodings, names = build_encoding(known_faces_dir)
    np.savez(ENCODINGS_CACHE, encodings = np.array(encodings), names = np.array(names))
    return encodings, names

def identify_face(encoding, known_encodings, known_names, tolerance=RECOGNITION_TOLERANCE): 
    #
    #the tolerance parameter can be adjusted to make the recognition more or less strict.
    #further use of this project will consist of data collection to determine optimal
    #STAGE AT THE MOMMENT: FUNCTIONAL 
    #

    if not known_encodings:
        return "Unknown"
    
    distances = face_recognition.face_distance(known_encodings, encoding)
    best_match_index = distances.argmin() #armin returns the index of the smallest distance
    if distances[best_match_index] < tolerance:
        return known_names[best_match_index]
    else:
        return "Unknown" 


def identify_in_frame(frame, known_encodings, known_names):
    #
    #function to be called in main loop to reset the name of the person in the frame.
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # convert to RGB for face_recognition
    locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    #
    #dealing with multiple faces present
    #
    known_here = [] # list of people in the frame
    saw_unknown = False

    for encoding in encodings:
        name = identify_face(encoding, known_encodings, known_names)
        if name.lower() == "unknown":
            saw_unknown = True
        elif name not in known_here:
            known_here.append(name) # CASE only add if not already present, could loop
    
    return known_here, saw_unknown 
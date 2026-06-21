#making
import os
import cv2
import face_recognition  # this relies on the dlib library, CPU intensive
# later to be revamped using the AI Camera 

"""
Loops through the known faces directory, based on the folder name. Encodes faces and store alongside
the name for comparison. 
"""
def load_known_faces(known_faces_dir):
    encodings = []
    names = []

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir): # looks for folder containing faces
            continue

        for filename in os.listdir(person_dir): # IGNORE no catch case for files not ending in .jpg or .png
            # catch case
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            # load the preset images and encode the faces
            path = os.path.join(person_dir, filename)  
            image = face_recognition.load_image_file(path) 
            face_encodings = face_recognition.face_encodings(image) 

            if not face_encodings:  # if no faces found, skip
                continue

            encodings.append(face_encodings[0])  
            """
            assume one face per image. UPDATE EVENTUALLY 
            """

            names.append(person_name) 

    return encodings, names

def identify_face(encoding, known_encodings, known_names, tolerance=0.5): 
    """
    the tolerance parameter can be adjusted to make the recognition more or less strict.
    further use of this project will consist of data collection to determine optimal
    STAGE AT THE MOMMENT: FUNCTIONAL 
    """
    
    if not known_encodings: # for unknown faces return "Unknown"
        return "Unknown"

    distances = face_recognition.face_distance(known_encodings, encoding)
    best_match_index = distances.argmin()  # get the index of the closest match

    if distances[best_match_index] <= tolerance:
        return known_names[best_match_index]
    else:
        return "Unknown"


def identify_in_frame(frame, known_encodings, known_names):
    """
    function to be called in main loop to reset the name of the person in the frame.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # convert to RGB for face_recognition
    locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    """
    dealing with multiple faces present
    """
    known_here = [] # list of people in the frame
    saw_unknown = False

    for encoding in encodings:
        name = identify_face(encoding, known_encodings, known_names)
        if name.lower() == "unknown":
            saw_unknown = True
        elif name not in known_here:
            known_here.append(name) # CASE only add if not already present, could loop
    
    return known_here, saw_unknown 
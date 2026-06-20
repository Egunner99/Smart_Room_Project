#making
import os
import face_recognition  # this relies on the dlib library, CPU intensive
# later to be revamped using the AI Camera 

"""
Loops through the known faces directory, based on the folder name. Encodes faces and store alongside
the name for comparison. 
"""
def load_known_faces(known_faces_dir):
    encodings = []
    names = []

    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir): # looks for folder containing faces
            continue

        for filename in os.listdir(person_dir): # no catch case for files not ending in .jpg or .png
         

    

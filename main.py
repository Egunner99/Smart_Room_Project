#revamped

import os
import time
import cv2
import numpy as np


from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

# load from other files UPDATE NEEDED POST ORG
from smart_room.recognizer import load_known_faces, identify_in_frame
from smart_room.action import handle_person
from smart_room.gestures import classify_gesture, handle_gesture
from smart_room.config import KNOWN_FACES_DIR, COOLDOWN_SECONDS, CAPTURES_DIR, POSE_MODEL, POSE_THRESHOLD, FACE_INTERVAL, GESTURE_COOLDOWN

IMG_SIZE = (480, 640)

os.makedirs(CAPTURES_DIR, exist_ok=True)  # ensure captures directory exists
os.makedirs(MODELS_DIR, exist_ok=True)  # ensure models directory exists

#load the pose model and set up camera, tested in gesture_control_test.py WOKRING
imx500 = IMX500(POSE_MODEL) 
intrinsics = imx500.network_intrinsics
if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "pose estimation"
if intrinsics.inference_rate is None:
    intrinsics.inference_rate = 10
intrinsics.update_with_defaults()

#bottleneck here SOLVED
print("faces loaded.") 
known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
print("known faces loaded")


# set and start the camera
picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12
)
picam2.start(config)
imx500.set_auto_aspect_ratio()
time.sleep(2)  # allow camera to warm up

print("the smart room is active. to cancel press ctrl+c or terminate the program")

# states to track
last_seen_time = {} # cooldown dictionary to track per person SOLVES MULTI
present_known = [] # help gesture control by known 
last_face_check = 0 # to limit face recognition frequency
last_gesture_check = None # to prevent repeated gesture triggers
last_gesture_time = {} # to track when each gesture was last triggered

# trigger time exist in a dictionary to differentiate between people


try:
    while True:
        request = picam2.capture_request()
        metadata = request.get_metadata()
        do_face = time.time() - last_face_check > FACE_INTERVAL
        frame = request.make_array("main") if do_face else None
        request.release()

        # face recognition loop CPU
        if do_face:
            now = time.time()
            last_face_check = now 
            known_here, saw_unknown = identify_in_frame(frame, known_encodings, known_names)
            present_known =  known_here

            names_here = set(known_here)
            if saw_unknown:
                names_here.add("Unknown")

            music_person = known_here[0] if len(known_here) > 0 else None

            for person in names_here:
                if now - last_seen_time.get(person, 0) < COOLDOWN_SECONDS:
                    last_seen_time[person] = now  # fix updateing time to prevent triggers
                    continue  

                last_seen_time[person] = now
                print(f"{person} entered the room")

                if person == "Unknown":
                    handle_person("Unknown", frame, play_music = (not known_here))
                else:
                    handle_person(person, frame, play_music = (person == music_person))
            

        # gesture recognition loop NPU, only if known person is present
        if present_known:
            np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
            if np_outputs is not None:
                keypoints, scores, boxes = postprocess_higherhrnet(
                    outputs=np_outputs,
                    img_size=IMG_SIZE,
                    img_w_pad=(0, 0), img_h_pad=(0, 0),
                    detection_threshold=POSE_THRESHOLD,
                    network_postprocess=True,
                )

                if scores is not None and len(scores) > 0:
                    kp = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
                    gesture = classify_gesture(kp[0])
                    if gesture != last_gesture_check: #acting occurs only if hand state changes
                        if gesture and gesture != "neutral": # ignore neutral gestures
                            print(gesture)
                            handle_gesture(gesture, present_known[0]) # handle first person detected
                            last_gesture_time[gesture] = time.time()  # update the last triggered time for this gesture
                        last_gesture_check = gesture # reset the gesture check to prevent repeated triggers
                else:
                    last_gesture_check = None
        else:
            last_gesture_check = None

        time.sleep(0.1)  # 10 frames per second

except KeyboardInterrupt:
    print("stopping smart room")

finally:
    picam2.stop()        


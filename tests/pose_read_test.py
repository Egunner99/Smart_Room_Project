#making
# source ~/smart_room_project/venv/bin/activate
# cd ~/picamera2/examples/imx500
# DISPLAY=:0 python3 imx500_pose_estimation_higherhrnet_demo.py \
#    --model /usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk
# 

"""
to run
cd ~/smart_room_project
DISPLAY=:0 python3 tests/pose_read_test.py
"""

import time
import numpy as np
from picamera2 import Picamera2

# NPU on the AI Camera # mimic form official demo -----------------------
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

MODEL = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
IMG_SIZE = (480, 640)

THRESHOLD = 0.5

L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10

MIN_CONFIDENCE = 0.5
RAISE_MARGIN = 40 #strict

def hand_raised(person):
    ls, lw = person[L_SHOULDER], person[L_WRIST]
    if ls[2]  > MIN_CONFIDENCE and lw[2] > MIN_CONFIDENCE and lw[1] < ls[1] - RAISE_MARGIN:
        return "left hand raised"
    rs, rw = person[R_SHOULDER], person[R_WRIST]
    if rs[2] > MIN_CONFIDENCE and rw[2] > MIN_CONFIDENCE and rw[1] < rs[1] - RAISE_MARGIN:
        return "right hand raised" 
    
    return None



imx500 = IMX500(MODEL)
intrinsics = imx500.network_intrinsics

if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "pose estimation"
if intrinsics.inference_rate is None:
    intrinsics.inference_rate = 10
intrinsics.update_with_defaults()

# still from the demo --------------------------    ---------------------------------------

picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(controls = {"FrameRate" : intrinsics.inference_rate}, buffer_count = 12)
imx500.show_network_fw_progress_bar()
# run with monitor
picam2.start(config, show_preview = True)
imx500.set_auto_aspect_ratio()

last_state = None #tracking the state to not have repeating 

try:
    while True:
        metadata = picam2.capture_metadata()
        np_outputs = imx500.get_outputs(metadata = metadata, add_batch = True)

        if np_outputs is None: # frames that didnt finsih processing yet
            continue

    
    # feed the raw outputs into the postprocessing function to get keypoints, scores, and bounding boxes
    
        keypoints, scores, boxes = postprocess_higherhrnet(
            outputs = np_outputs,
            img_size = IMG_SIZE,
            img_w_pad = (0,0), img_h_pad = (0,0),
            detection_threshold = THRESHOLD,
            network_postprocess = True
        )

        if scores is None or len(scores) == 0:
            last_state = None
            continue

    # stack everyone into one (people, 17, 3) grid so person[9] is always the same joint
        kp = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17,3))
        person = kp[0] 
        ls, rs, lw, rw = person[L_SHOULDER], person[R_SHOULDER], person[L_WRIST], person[R_WRIST]

        raised = hand_raised(person)
        # print("state: ", raised) #debug line
        if raised:
            if raised != last_state:
                print(raised)
            last_state = raised


except KeyboardInterrupt:
    print("exiting pose estimation test")
finally:
    picam2.stop()
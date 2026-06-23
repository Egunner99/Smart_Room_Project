# working

import csv 
import time
import numpy as np
import sys

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

from smart_room.gestures import extract_features
from smart_room.config import POSE_MODEL, POSE_THRESHOLD

IMG_SIZE = (480, 640)
SAMPLES_PER_RUN = 100
DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else "gesture_data.csv"  

imx500 = IMX500(POSE_MODEL)
intrinsics = imx500.network_intrinsics
if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "pose estimation"
if intrinsics.inference_rate is None:
    intrinsics.inference_rate = 10
intrinsics.update_with_defaults()

picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12
)
picam2.start(config, show_preview=True) #assist with debugging 
#proper pose for gesture but failing, visual helps # MONITOR CONNECTED TO PI NEEDED
imx500.set_auto_aspect_ratio()

# capture a frames worth and extract features
def read_features():
    metadata = picam2.capture_metadata()
    np_outputs = imx500.get_outputs(metadata = metadata, add_batch = True)
    if np_outputs is None:
        print("bug: no outputs from imx500") # debugging
        return None
    keypoints, scores, boxes = postprocess_higherhrnet(
        outputs=np_outputs, 
        img_size=IMG_SIZE,
        img_w_pad = (0,0),
        img_h_pad = (0,0),
        detection_threshold=POSE_THRESHOLD,
        network_postprocess = True
    )

    if scores is None or len(scores) == 0:
        print("bug: no valid detections") # debugging
        return None
    kp = np.reshape(np.stack(keypoints, axis=0), (len(keypoints), 17,3))
    features = extract_features(kp[0])  # only take the first person detected
    if features is None:
        print("bug: features are None not confident shoulders") # debugging
        return None
    return features

def record_gesture(label):
    print(f"get into '{label}' pose")
    time.sleep(3)
    print("recording")
    count = 0
    misses = 0
    with open(DATA_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        while count < SAMPLES_PER_RUN:
            features = read_features()
            if features is None:
                misses += 1
                if misses % 30 == 0:
                    print("not clear pose, fix your position")
                continue
            writer.writerow([label] + features)
            count += 1
            if count % 25 == 0:
                print(f"{count} / {SAMPLES_PER_RUN} samples recorded")
    print(f"done recording '{label}' pose, {misses} misses")
                
        


try:
    while True:
        label = input("\n label (neutral/left_hand/right_hand/both_hands/t_pose) or q to quit: ").strip().lower()
        if label == "q":
            break
        if label:
            record_gesture(label)
finally:
    picam2.stop()
    print("data saved to", DATA_FILE)


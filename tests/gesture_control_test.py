#testing isolated

import time
import numpy as np

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

from smart_room.gestures import hand_raised, handle_gesture

MODEL = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
IMG_SIZE = (480, 640)
THRESHOLD = 0.5

imx500 = IMX500(MODEL)
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
imx500.show_network_fw_progress_bar()
picam2.start(config, show_preview=True)
imx500.set_auto_aspect_ratio()

print("gesture test... right hand = next, left hand = play/pause. ctrl+c to stop")

last_state = None

try:
    while True:
        metadata = picam2.capture_metadata()
        np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
        if np_outputs is None:
            continue

        keypoints, scores, boxes = postprocess_higherhrnet(
            outputs=np_outputs,
            img_size=IMG_SIZE,
            img_w_pad=(0, 0), img_h_pad=(0, 0),
            detection_threshold=THRESHOLD,
            network_postprocess=True,
        )

        if scores is None or len(scores) == 0:
            last_state = None
            continue

        kp = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
        person = kp[0]

        raised = hand_raised(person)
        if raised != last_state:
            if raised:
                print(f"{raised} hand raised")
                handle_gesture(raised)
            last_state = raised

except KeyboardInterrupt:
    print("stopping gesture control test")
finally:
    picam2.stop()
from picamera2 import Picamera2
import cv2
import time
import os

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()
time.sleep(1)

name = input("folder name (e.g. gio): ").strip()
save_dir = f"known_faces/{name}"
os.makedirs(save_dir, exist_ok=True)

total = 20
delay = 5

print(f"starting in 5 seconds, get in position")
time.sleep(5)

for i in range(total):
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    filename = f"{save_dir}/{name}_{i}.jpg"
    cv2.imwrite(filename, frame_bgr)
    print(f"saved {filename} ({i+1}/{total})")

    if i < total - 1:
        time.sleep(delay)

picam2.stop()
print("done")

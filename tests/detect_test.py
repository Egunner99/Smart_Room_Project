from picamera2 import Picamera2
import face_recognition
import cv2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

print("running... ctrl+c to stop")

face_present = False

try:
    while True:
        frame = picam2.capture_array()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_recognition.face_locations(rgb_frame)

        if faces and not face_present:
            print(f"face detected - {len(faces)} found")
            cv2.imwrite("last_detection.jpg", frame)
            face_present = True
        elif not faces:
            face_present = False

        time.sleep(0.5)

except KeyboardInterrupt:
    print("stopping")

finally:
    picam2.stop()

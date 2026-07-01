import cv2
import os
import time

output_dir = "pose_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

fps_target = 5
frame_interval = 1.0 / fps_target

frame_count = 0
next_capture_time = time.perf_counter()

print("Recording... Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.perf_counter()

        if now >= next_capture_time:
            filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(filename, frame)

            print(f"Saved {filename}")

            frame_count += 1
            next_capture_time += frame_interval

except KeyboardInterrupt:
    print("Stopped.")

cap.release()
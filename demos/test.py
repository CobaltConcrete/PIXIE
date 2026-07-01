"""
Minimal webcam -> MJPEG HTTP stream test.
Open http://localhost:8080/ in your browser to see the webcam feed.
Ctrl+C to stop.
"""
import cv2
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── shared state ──────────────────────────────────────────────────────────────
_frame: bytes = b""
_lock = threading.Lock()
_stop = threading.Event()

# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == "/":
            body = (
                b"<html><body style='margin:0;background:#000'>"
                b"<img src='/stream' style='max-width:100%;height:auto'>"
                b"</body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while not _stop.is_set():
                    with _lock:
                        frame = _frame
                    if frame:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + frame + b"\r\n"
                        )
                    time.sleep(0.033)  # ~30fps
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

# ── start server in background ────────────────────────────────────────────────
server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
t = threading.Thread(target=server.serve_forever, daemon=True)
t.start()
print("Server started -> http://localhost:8080/")

# ── webcam loop ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
if not cap.isOpened():
    print("ERROR: could not open /dev/video0")
    print("Check: ls /dev/video*")
    exit(1)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam opened: {w}x{h}")
print("Open http://localhost:8080/ in your browser")
print("Ctrl+C to stop")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: failed to read frame")
            break

        # draw a frame counter so we can tell it's live
        cv2.putText(frame, f"LIVE TEST", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _frame = buf.tobytes()

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    _stop.set()
    cap.release()
    server.shutdown()
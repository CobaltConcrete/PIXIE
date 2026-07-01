import os, sys, time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg
from pixielib.datasets import detectors


# ══════════════════════════════════════════════════════════════════════════════
# MJPEG HTTP streamer  (headless-friendly live preview)
# ══════════════════════════════════════════════════════════════════════════════

class MJPEGStreamer:
    """
    Serves rendered frames as an MJPEG stream over HTTP so they can be watched
    in any browser — no display server required.

    Open  http://<host>:<port>/  in your browser to watch.
    """

    def __init__(self, port: int = 8080, jpeg_quality: int = 85):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.port    = port
        self.quality = jpeg_quality
        self._frame: bytes = b""          # latest JPEG-encoded frame
        self._lock   = threading.Lock()
        self._stop   = threading.Event()

        streamer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):   # silence access log
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
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame"
                    )
                    self.end_headers()
                    try:
                        while not streamer._stop.is_set():
                            with streamer._lock:
                                frame = streamer._frame
                            if frame:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                                    + frame + b"\r\n"
                                )
                            time.sleep(0.01)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

        self._server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[stream] Preview -> http://localhost:{port}/  (or your server IP)")

    def push(self, rgb_np: np.ndarray):
        """Encode an RGB uint8 frame and make it available to connected clients."""
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.quality],
        )
        if ok:
            with self._lock:
                self._frame = buf.tobytes()

    def stop(self):
        self._stop.set()
        self._server.shutdown()


# ══════════════════════════════════════════════════════════════════════════════
# RTSP streamer  (ffmpeg subprocess)
# ══════════════════════════════════════════════════════════════════════════════

class RTSPStreamer:
    """
    Pushes rendered frames into a local RTSP server via ffmpeg.

    Requires ffmpeg with libx264.  Start a local RTSP server first, e.g.:
        docker run --rm -it -p 8554:8554 aler9/rtsp-simple-server
    Then connect any RTSP player (VLC, ffplay) to:
        rtsp://localhost:8554/live
    """

    def __init__(self, port: int = 8554, fps: float = 25.0):
        import subprocess, shutil
        self.fps  = fps
        self.port = port
        self._proc = None
        self._size = None
        self._ffmpeg = shutil.which("ffmpeg")
        if self._ffmpeg is None:
            raise RuntimeError("ffmpeg not found in PATH — required for RTSP streaming.")

    def _start(self, w: int, h: int):
        import subprocess
        cmd = [
            self._ffmpeg, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-vcodec", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "rtsp", f"rtsp://localhost:{self.port}/live",
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._size = (w, h)
        print(f"[stream] RTSP -> rtsp://localhost:{self.port}/live")
        print(f"[stream] Watch with:  ffplay rtsp://localhost:{self.port}/live")

    def push(self, rgb_np: np.ndarray):
        h, w = rgb_np.shape[:2]
        if self._proc is None:
            self._start(w, h)
        bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        try:
            self._proc.stdin.write(bgr.tobytes())
        except BrokenPipeError:
            pass

    def stop(self):
        if self._proc and self._proc.stdin:
            self._proc.stdin.close()
            self._proc.wait()


# ══════════════════════════════════════════════════════════════════════════════
# FFmpeg-based video writer (H.264 / yuv420p mp4 - playable in VS Code)
# ══════════════════════════════════════════════════════════════════════════════

class FFmpegVideoWriter:
    """
    Writes RGB uint8 frames to an H.264-encoded mp4 via an ffmpeg subprocess.
    cv2.VideoWriter's mp4v fourcc often produces files VS Code's built-in
    player can't decode; libx264 + yuv420p is broadly compatible.
    """

    def __init__(self, out_path: str, fps: float, width: int, height: int):
        import subprocess, shutil
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found in PATH — required for video output.")

        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-an",
            "-vcodec", "libx264", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def write(self, rgb_np: np.ndarray):
        try:
            self._proc.stdin.write(rgb_np.tobytes())
        except BrokenPipeError:
            pass

    def release(self):
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()



def open_capture(input_arg: str) -> cv2.VideoCapture:
    """
    Open a cv2.VideoCapture for:
      - webcam index (e.g. "0", "1")  -> opened via V4L2 (Linux/WSL/Docker friendly)
      - video file path                -> opened via FFMPEG backend
      - stream URL (rtsp://, http://, etc.) -> opened via FFMPEG backend
    """
    if input_arg.isdigit():
        # /dev/video{N} via V4L2 - required for webcams passed into Docker/WSL containers
        device_path = f"/dev/video{input_arg}"
        cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            # fallback: try plain index with V4L2 backend
            cap = cv2.VideoCapture(int(input_arg), cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(input_arg, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open input source: {input_arg}\n"
            f"If this is a webcam, check that /dev/video* is passed into the "
            f"container (e.g. `--device /dev/video0`) and that you're in the "
            f"`video` group."
        )

    if input_arg.isdigit():
        # FHD webcam over V4L2 defaults to slow YUYV; force MJPG which the
        # device supports at full frame rate.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

    return cap


def is_image_path(path: str) -> bool:
    return os.path.splitext(path)[-1].lower() in [
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    ]


def auto_output_path(input_arg: str, model_name: str, is_video: bool) -> str:
    if input_arg.isdigit():
        source_name = f"webcam{input_arg}"
    else:
        source_name = os.path.splitext(os.path.basename(input_arg))[0]
    ext = "mp4" if is_video else "png"
    out_dir = os.path.join("outputs", source_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{source_name}_{model_name}_pose.{ext}")


# ══════════════════════════════════════════════════════════════════════════════
# Frame preprocessing -> PIXIE batch dict
# ══════════════════════════════════════════════════════════════════════════════

def frame_to_batch(frame_bgr: np.ndarray, detector, crop_size: int = 224,
                    scale: float = 1.1, device: str = "cuda:0", render_size: int = 512):
    """
    Convert a raw BGR frame into a PIXIE-compatible batch dict (single image),
    mirroring TestData's preprocessing (crop around detected bbox, resize,
    normalize to [0,1]).

    Returns None if no person is detected.
    """
    from pixielib.utils import util as pixie_util
    from torchvision import transforms

    # Fix 1: handle grayscale BEFORE converting to tensor
    if frame_bgr.ndim == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    elif frame_bgr.shape[2] == 1:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape

    # Fix 2: wrap tensor in a list — FasterRCNN expects List[Tensor]
    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float()
    bbox = detector.run(image_tensor)  # <-- list wrapping here
    if bbox is None:
        return None

    left, top, right, bottom = bbox
    old_size = max(right - left, bottom - top)
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)

    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2]
    ])
    DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])

    from skimage.transform import estimate_transform, warp
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    image_norm = image_rgb / 255.0
    dst_image = warp(image_norm, tform.inverse, output_shape=(crop_size, crop_size))
    dst_image = dst_image.transpose(2, 0, 1)

    # HD image — resized to render_size square to match renderer output dimensions
    image_hd_sq = cv2.resize(image_rgb, (render_size, render_size)).astype(np.float64) / 255.0
    image_hd = image_hd_sq.transpose(2, 0, 1)

    batch = {
        'image': torch.tensor(dst_image).float()[None, ...],
        'image_hd': torch.tensor(image_hd).float()[None, ...],
        'name': 'frame',
        'imagename': 'frame',
        'tform': torch.tensor(tform.params).float().T[None, ...],
        'original_image': torch.tensor(image_norm.transpose(2, 0, 1)).float(),
    }
    pixie_util.move_dict_to_device(batch, device)
    return batch


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    device = args.device

    if not torch.cuda.is_available():
        print('CUDA is not available! use CPU instead')
        device = 'cpu'
    else:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    # -- load model
    pixie_cfg.model.use_tex = args.useTex
    pixie = PIXIE(config=pixie_cfg, device=device)
    visualizer = Visualizer(render_size=args.render_size, config=pixie_cfg,
                             device=device, rasterizer_type=args.rasterizer_type)

    # face/body detector used for per-frame cropping
    detector = detectors.FasterRCNN(device=device)

    # -- determine input mode
    is_image = is_image_path(args.input) and os.path.exists(args.input)

    streamer = None
    if args.stream is not None:
        if args.stream == "mjpeg":
            streamer = MJPEGStreamer(port=8080)
        elif args.stream == "rtsp":
            streamer = RTSPStreamer(port=8554)
        elif args.stream == "window":
            cv2.namedWindow("pose_estimator", cv2.WINDOW_NORMAL)

    writer = None
    out_path = None
    if args.output is not None:
        model_name = args.rasterizer_type
        out_path = (args.output if args.output != "__auto__"
                     else auto_output_path(args.input, model_name, is_video=not is_image))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    try:
        if is_image:
            # single image -> single rendered frame
            frame_bgr = cv2.imread(args.input)
            rendered = process_frame(frame_bgr, pixie, visualizer, detector,
                                      args.rasterizer_type, args.render_size, device)
            if rendered is not None:
                if streamer is not None:
                    push_to_stream(streamer, rendered, args.stream)
                if out_path is not None:
                    cv2.imwrite(out_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
                    print(f'-- saved {out_path}')
            else:
                print('-- no person detected in image')
        else:
            # video / webcam / RTSP stream -> continuous loop
            cap = open_capture(args.input)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            print('-- starting live loop, press Ctrl+C to stop'
                  + (' (or q in window mode)' if args.stream == 'window' else ''))

            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    print('-- end of stream / cannot read frame')
                    break

                rendered = process_frame(frame_bgr, pixie, visualizer, detector,
                                          args.rasterizer_type, args.render_size, device)
                if rendered is None:
                    # no person detected this frame -> show raw frame instead
                    rendered = cv2.cvtColor(
                        cv2.resize(frame_bgr, (args.render_size, args.render_size)),
                        cv2.COLOR_BGR2RGB
                    )

                if streamer is not None:
                    if push_to_stream(streamer, rendered, args.stream) is False:
                        break  # 'q' pressed in window mode

                if out_path is not None:
                    if writer is None:
                        h, w = rendered.shape[:2]
                        writer = FFmpegVideoWriter(out_path, fps, w, h)
                    writer.write(rendered)

            cap.release()
    finally:
        if writer is not None:
            writer.release()
            print(f'-- saved {out_path}')
        if streamer is not None:
            streamer.stop()
        if args.stream == "window":
            cv2.destroyAllWindows()


def process_frame(frame_bgr, pixie, visualizer, detector, rasterizer_type, render_size, device):
    """Run PIXIE on a single BGR frame and return an RGB uint8 rendered image, or None."""
    batch = frame_to_batch(frame_bgr, detector, device=device, render_size=render_size)
    if batch is None:
        return None

    data = {'body': batch}
    param_dict = pixie.encode(data)
    codedict = param_dict['body']
    moderator_weight = param_dict['moderator_weight']

    opdict = pixie.decode(codedict, param_type='body')

    if rasterizer_type == 'standard':
        tform = batch['tform']
        tform = torch.inverse(tform).transpose(1, 2)
        original_image = batch['original_image'][None, ...]
        visualizer.recover_position(opdict, batch, tform, original_image)

    visdict = visualizer.render_results(opdict, batch['image_hd'],
                                         moderator_weight=moderator_weight, overlay=True)

    shape_img = visdict['color_shape_images'][0]
    rendered = shape_img.detach().cpu().numpy().transpose(1, 2, 0)
    rendered = np.clip(rendered, 0, 1)
    rendered = (rendered * 255).astype(np.uint8)
    rendered = cv2.resize(rendered, (render_size, render_size))
    return rendered


def push_to_stream(streamer, rendered_rgb, mode):
    """Push a rendered RGB frame to the active streamer. Returns False if user quit (window mode)."""
    if mode == "mjpeg":
        streamer.push(rendered_rgb)
    elif mode == "rtsp":
        streamer.push(rendered_rgb)
    elif mode == "window":
        bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("pose_estimator", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXIE live pose estimator')

    parser.add_argument("--input", required=True,
                         help=(
                             "Input - one of:\n"
                             "  image path  (.jpg/.png/...)\n"
                             "  video path  (.mp4/.avi/...)\n"
                             "  stream URL  (rtsp://...)\n"
                             "  webcam index (0, 1, ...)"
                         ))
    parser.add_argument(
        "--output",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="PATH",
        help=(
            "Output path (optional):\n"
            "  omitted          -> do not save output\n"
            "  --output         -> auto-name: ./outputs/{source}/{source}_{model_name}_pose.png/mp4\n"
            "  --output PATH    -> save to PATH"
        ),
    )
    parser.add_argument(
        "--stream",
        nargs="?",
        const="mjpeg",
        default=None,
        metavar="MODE",
        choices=["mjpeg", "window", "rtsp"],
        help=(
            "Live preview mode (video/stream input only):\n"
            "  mjpeg   HTTP MJPEG on :8080 - open in any browser\n"
            "  window  cv2.imshow - needs a display (X11/VNC)\n"
            "  rtsp    push to RTSP via ffmpeg (needs rtsp-simple-server)\n"
            "Omit value to default to mjpeg."
        ),
    )
    parser.add_argument('--device', default='cuda:0', type=str,
                         help='set device, cpu for using cpu')
    parser.add_argument('--render_size', default=512, type=int,
                         help='image size of renderings')
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                         help='rasterizer type: pytorch3d or standard')
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                         help='whether to use FLAME texture model for albedo')

    main(parser.parse_args())
import os, sys, time
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2
import imageio
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg


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
        from http.server import BaseHTTPRequestHandler, HTTPServer

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
                    # Tiny HTML page that auto-loads the stream
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

        self._server = HTTPServer(("0.0.0.0", port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[stream] Preview → http://localhost:{port}/  (or your server IP)")

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
# Webcam capture via V4L2
# ══════════════════════════════════════════════════════════════════════════════

class WebcamCapture:
    """
    Thread-safe webcam reader using OpenCV's V4L2 backend (CAP_V4L2).

    Frames are grabbed on a background thread so the main loop never blocks
    waiting for the next sensor exposure.  Call .read() to get the most recent
    BGR frame, or .read_rgb() for an RGB array ready for PIXIE.

    Parameters
    ----------
    device_index : int
        V4L2 device index — 0 means /dev/video0, 1 means /dev/video1, etc.
    width, height : int
        Requested capture resolution.  The driver may round to the nearest
        supported mode; call .actual_size() to check what was negotiated.
    fps : int
        Requested frame rate.
    mjpeg : bool
        If True, ask the driver to use the MJPEG pixel format before decoding.
        Most USB cameras stream faster this way at high resolutions.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        mjpeg: bool = True,
    ):
        self._cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"[webcam] Cannot open /dev/video{device_index} via V4L2. "
                "Check that the device exists and you have read permission."
            )

        # Optionally switch to MJPEG for higher-bandwidth efficiency
        if mjpeg:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS,          fps)

        # Report what the driver actually negotiated
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_f = self._cap.get(cv2.CAP_PROP_FPS)
        print(
            f"[webcam] /dev/video{device_index}  "
            f"{actual_w}×{actual_h} @ {actual_f:.0f} fps"
            f"{'  (MJPEG)' if mjpeg else ''}"
        )

        # Prime the ring buffer with one frame
        self._frame: np.ndarray | None = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()

        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

        # Block until the first frame arrives (max 5 s)
        deadline = time.time() + 5.0
        while self._frame is None and time.time() < deadline:
            time.sleep(0.01)
        if self._frame is None:
            raise RuntimeError("[webcam] Timed out waiting for first frame.")

    # ------------------------------------------------------------------
    def _grab_loop(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame          # BGR uint8

    def read(self) -> np.ndarray:
        """Return the latest BGR frame (H×W×3 uint8)."""
        with self._lock:
            return self._frame.copy()

    def read_rgb(self) -> np.ndarray:
        """Return the latest RGB frame (H×W×3 uint8)."""
        return cv2.cvtColor(self.read(), cv2.COLOR_BGR2RGB)

    def actual_size(self) -> tuple[int, int]:
        """Return (width, height) as negotiated with the driver."""
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _frame_from_webcam(webcam: WebcamCapture, device: str, pixie_cfg, iscrop: bool) -> dict:
    """
    Grab one RGB frame from the webcam, wrap it in the dict structure that
    PIXIE's TestData would normally produce, and move it to *device*.

    We re-use TestData's internal crop / normalise logic by writing the frame
    to a temporary file and constructing a one-shot dataset — this keeps all
    preprocessing identical to the file-based path.
    """
    import tempfile
    rgb = webcam.read_rgb()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, bgr)

    try:
        ds    = TestData(tmp_path, iscrop=iscrop, body_detector='rcnn')
        batch = ds[0]
    finally:
        os.unlink(tmp_path)

    util.move_dict_to_device(batch, device)
    batch['image']    = batch['image'].unsqueeze(0)
    batch['image_hd'] = batch['image_hd'].unsqueeze(0)
    return batch


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    savefolder = args.savefolder
    device     = args.device
    os.makedirs(savefolder, exist_ok=True)

    # ── CUDA check ────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print('CUDA is not available! using CPU instead')
    else:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled       = True

    # ── Optional MJPEG HTTP streamer ──────────────────────────────────────────
    streamer: MJPEGStreamer | None = None
    if args.stream:
        streamer = MJPEGStreamer(port=args.stream_port, jpeg_quality=args.stream_quality)

    # ── Optional webcam source for pose sequence ───────────────────────────────
    webcam: WebcamCapture | None = None
    if args.webcam:
        webcam = WebcamCapture(
            device_index=args.webcam_device,
            width=args.webcam_width,
            height=args.webcam_height,
            fps=args.webcam_fps,
            mjpeg=args.webcam_mjpeg,
        )

    # ── Load models ───────────────────────────────────────────────────────────
    pixie_cfg.model.use_tex = args.useTex
    pixie       = PIXIE(config=pixie_cfg, device=device)
    visualizer  = Visualizer(
        render_size=args.render_size,
        config=pixie_cfg,
        device=device,
        rasterizer_type=args.rasterizer_type,
    )

    # ── 1. Fit SMPL-X from the input identity image ───────────────────────────
    testdata = TestData(args.inputpath, iscrop=args.iscrop, body_detector='rcnn')
    batch    = testdata[0]
    util.move_dict_to_device(batch, device)
    batch['image']    = batch['image'].unsqueeze(0)
    batch['image_hd'] = batch['image_hd'].unsqueeze(0)
    name        = batch['name']
    input_image = batch['image']
    data        = {'body': batch}

    param_dict      = pixie.encode(data)
    input_codedict  = param_dict['body']
    input_opdict    = pixie.decode(input_codedict, param_type='body')
    input_opdict['albedo'] = visualizer.tex_flame2smplx(input_opdict['albedo'])
    visdict         = visualizer.render_results(input_opdict, data['body']['image_hd'], overlay=True)
    input_image     = batch['image_hd'].clone()
    input_shape     = visdict['shape_images'].clone()

    # ── 2. Pose / expression source ───────────────────────────────────────────
    os.makedirs(os.path.join(savefolder, name), exist_ok=True)

    if webcam:
        # ── 2a. Live webcam loop ───────────────────────────────────────────────
        print("[webcam] Starting live animation loop — press Ctrl-C to stop.")
        frame_idx = 0
        try:
            while True:
                pose_batch = _frame_from_webcam(webcam, device, pixie_cfg, args.iscrop)
                data       = {'body': pose_batch}

                param_dict      = pixie.encode(data)
                codedict        = param_dict['body']
                moderator_weight = param_dict['moderator_weight']
                opdict          = pixie.decode(codedict, param_type='body')

                if args.reproject_mesh and args.rasterizer_type == 'standard':
                    tform          = pose_batch['tform'][None, ...]
                    tform          = torch.inverse(tform).transpose(1, 2)
                    original_image = pose_batch['original_image'][None, ...]
                    visualizer.recover_position(opdict, pose_batch, tform, original_image)

                visdict        = visualizer.render_results(
                    opdict, data['body']['image_hd'],
                    moderator_weight=moderator_weight, overlay=True,
                )
                pose_ref_shape = visdict['color_shape_images'].clone()

                # Transfer identity → pose frame
                for param in ['shape', 'tex', 'body_cam', 'light']:
                    codedict[param] = input_codedict[param]
                opdict = pixie.decode(codedict, param_type='body')
                opdict['albedo'] = input_opdict['albedo']
                visdict = visualizer.render_results(opdict, input_image)
                transfered_shape = visdict['shape_images'].clone()

                grid_visdict = {
                    'input':           input_image,
                    'rec':             input_shape,
                    'transfer':        transfered_shape,
                    'pose_ref':        pose_batch['image_hd'],
                    'pose_ref_shape':  pose_ref_shape,
                }
                grid_image_all = visualizer.visualize_grid(grid_visdict, size=512)

                if args.saveImages:
                    cv2.imwrite(
                        os.path.join(savefolder, name, f'{name}_animate_{frame_idx:05}.jpg'),
                        grid_image_all,
                    )

                # Push to MJPEG streamer (expects RGB)
                if streamer is not None:
                    streamer.push(grid_image_all[:, :, [2, 1, 0]])

                frame_idx += 1

        except KeyboardInterrupt:
            print("\n[webcam] Stopped by user.")
        finally:
            webcam.release()

    else:
        # ── 2b. File-based animation (original behaviour) ─────────────────────
        posedata = TestData(args.posepath, iscrop=args.iscrop, body_detector='rcnn')
        writer   = imageio.get_writer(
            os.path.join(savefolder, 'animation.gif'), mode='I'
        ) if args.saveGif else None

        for i, batch in enumerate(tqdm(posedata, dynamic_ncols=True)):
            if i % 1 == 0:
                util.move_dict_to_device(batch, device)
                batch['image']    = batch['image'].unsqueeze(0)
                batch['image_hd'] = batch['image_hd'].unsqueeze(0)
                data = {'body': batch}

                param_dict       = pixie.encode(data)
                codedict         = param_dict['body']
                moderator_weight = param_dict['moderator_weight']
                opdict           = pixie.decode(codedict, param_type='body')

                if args.reproject_mesh and args.rasterizer_type == 'standard':
                    tform          = batch['tform'][None, ...]
                    tform          = torch.inverse(tform).transpose(1, 2)
                    original_image = batch['original_image'][None, ...]
                    visualizer.recover_position(opdict, batch, tform, original_image)

                visdict        = visualizer.render_results(
                    opdict, data['body']['image_hd'],
                    moderator_weight=moderator_weight, overlay=True,
                )
                pose_ref_shape = visdict['color_shape_images'].clone()

                # Transfer identity → pose frame
                for param in ['shape', 'tex', 'body_cam', 'light']:
                    codedict[param] = input_codedict[param]
                opdict = pixie.decode(codedict, param_type='body')
                opdict['albedo'] = input_opdict['albedo']
                visdict = visualizer.render_results(opdict, input_image)
                transfered_shape = visdict['shape_images'].clone()

                grid_visdict = {
                    'input':          input_image,
                    'rec':            input_shape,
                    'transfer':       transfered_shape,
                    'pose_ref':       batch['image_hd'],
                    'pose_ref_shape': pose_ref_shape,
                }
                grid_image_all = visualizer.visualize_grid(grid_visdict, size=512)
                cv2.imwrite(
                    os.path.join(savefolder, name, f'{name}_animate_{i:05}.jpg'),
                    grid_image_all,
                )

                if writer is not None:
                    writer.append_data(grid_image_all[:, :, [2, 1, 0]])

                # Push to MJPEG streamer (expects RGB)
                if streamer is not None:
                    streamer.push(grid_image_all[:, :, [2, 1, 0]])

        if writer is not None:
            writer.close()

    if streamer is not None:
        streamer.stop()

    print(f'-- please check the results in {savefolder}')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXIE')

    # ── Identity image ────────────────────────────────────────────────────────
    parser.add_argument('-i', '--inputpath',
        default='TestSamples/body/woman-in-white-dress-3830468.jpg', type=str,
        help='Path to the identity image (file, folder, list, or video).')

    # ── Pose source: file-based (original) ────────────────────────────────────
    parser.add_argument('-p', '--posepath',
        default='TestSamples/animation', type=str,
        help='Path to pose-reference frames (folder / image list / video). '
             'Ignored when --webcam is set.')

    # ── Pose source: webcam (V4L2) ────────────────────────────────────────────
    parser.add_argument('--webcam', action='store_true',
        help='Use a live webcam as the pose source instead of --posepath.')
    parser.add_argument('--webcam-device', default=0, type=int, metavar='N',
        help='V4L2 device index (default: 0 → /dev/video0).')
    parser.add_argument('--webcam-width',  default=1280, type=int,
        help='Requested capture width  (default: 1280).')
    parser.add_argument('--webcam-height', default=720,  type=int,
        help='Requested capture height (default:  720).')
    parser.add_argument('--webcam-fps',    default=30,   type=int,
        help='Requested capture frame rate (default: 30).')
    parser.add_argument('--webcam-mjpeg',
        default=True, type=lambda x: x.lower() in ['true', '1'],
        help='Ask the V4L2 driver to use MJPEG encoding (default: True). '
             'Reduces USB bandwidth at high resolutions.')

    # ── MJPEG HTTP stream ─────────────────────────────────────────────────────
    parser.add_argument('--stream', action='store_true',
        help='Serve rendered frames as an MJPEG stream over HTTP '
             '(viewable in any browser without a display server).')
    parser.add_argument('--stream-port',    default=8080, type=int,
        help='TCP port for the MJPEG HTTP server (default: 8080).')
    parser.add_argument('--stream-quality', default=85,   type=int,
        help='JPEG quality for the stream, 1-100 (default: 85).')

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation', type=str,
        help='Output directory for results.')
    parser.add_argument('--device', default='cuda:0', type=str,
        help='Compute device ("cuda:0", "cpu", …).')

    # ── Pre-processing ────────────────────────────────────────────────────────
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
        help='Crop the input image before fitting (set False for pre-cropped images).')

    # ── Rendering ─────────────────────────────────────────────────────────────
    parser.add_argument('--render_size', default=1024, type=int,
        help='Render resolution (default: 1024).')
    parser.add_argument('--rasterizer_type', default='standard', type=str,
        help='"pytorch3d" or "standard".')
    parser.add_argument('--reproject_mesh',
        default=False, type=lambda x: x.lower() in ['true', '1'],
        help='Reproject mesh back to original image space '
             '(standard rasterizer only).')

    # ── Texture / DECA ────────────────────────────────────────────────────────
    parser.add_argument('--deca_path', default=None, type=str,
        help='Absolute path to DECA (enables facial details).')
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
        help='Use FLAME texture model (requires downloaded texture model).')
    parser.add_argument('--uvtex_type', default='SMPLX', type=str,
        help='"SMPLX" or "FLAME".')

    # ── Save options ──────────────────────────────────────────────────────────
    parser.add_argument('--saveVis',    default=True,  type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--saveGif',    default=True,  type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--saveObj',    default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--saveParam',  default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--savePred',   default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'])

    main(parser.parse_args())
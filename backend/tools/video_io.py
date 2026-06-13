import queue
import subprocess
import threading

import numpy as np

from .ffmpeg_cli import FFmpegCLI


class FramePrefetcher:
    """Decode frames on a background thread so I/O overlaps model inference.

    Interface-compatible with cv2.VideoCapture (read/release).
    """

    def __init__(self, video_cap, buffer_size=10):
        self.cap = video_cap
        self._buffer = queue.Queue(maxsize=buffer_size)
        self._stopped = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            self._buffer.put((ret, frame))
            if not ret:
                break

    def read(self):
        """Read the next frame; same contract as cv2.VideoCapture.read()."""
        return self._buffer.get()

    def get(self, propId):
        return self.cap.get(propId)

    def stop(self):
        """Stop prefetching without releasing the underlying video_cap."""
        self._stopped = True
        try:
            while not self._buffer.empty():
                self._buffer.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5)

    def release(self):
        self.stop()
        self.cap.release()


class FFmpegVideoWriter:
    """Write frames through an FFmpeg pipe using libx264 encoding.

    Interface-compatible with cv2.VideoWriter (write/release).
    """

    def __init__(self, output_path, fps, size):
        w, h = size
        cmd = [
            FFmpegCLI.instance().ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'fast',
            '-loglevel', 'error',
            output_path
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame):
        """Write one frame (a numpy BGR array)."""
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass

    def release(self):
        """Close the pipe and wait for encoding to finish."""
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self._process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)


def make_video_writer(output_path, fps, size):
    """Return an x264 FFmpeg writer, falling back to OpenCV's mp4v writer.

    Both returned objects expose the same ``write(frame)`` / ``release()``
    interface, so callers do not need to know which backend was chosen.

    Odd width/height go straight to OpenCV: libx264 with yuv420p requires even
    dimensions and would otherwise fail *asynchronously* inside the ffmpeg
    process (producing a silent 0-byte file that the try/except cannot catch),
    whereas OpenCV's mp4v writer handles odd dimensions fine.
    """
    import cv2

    width, height = size
    if width % 2 == 0 and height % 2 == 0:
        try:
            return FFmpegVideoWriter(output_path, fps, size)
        except Exception:
            pass
    return cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )

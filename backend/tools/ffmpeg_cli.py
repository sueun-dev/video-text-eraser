"""Locates a working FFmpeg binary for the current platform.

Resolution order: a binary that actually runs on this machine wins —
the bundled one first, then PATH, then the imageio-ffmpeg wheel. The bundled
macOS binary is x86-64 only, so Apple Silicon machines rely on the fallbacks.
"""

import os
import platform
import shutil
import stat
import subprocess
from functools import cached_property
from typing import Optional

from backend.config import BASE_DIR

from .common_tools import merge_big_file_if_not_exists


def _runs(binary: Optional[str]) -> bool:
    """Whether the binary exists and executes on this CPU/OS."""
    if not binary or not os.path.exists(binary):
        return False
    try:
        subprocess.run(
            [binary, "-version"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10, check=True,
        )
        return True
    except Exception:
        return False


class FFmpegCLI:
    """Singleton resolving the FFmpeg executable path once per process."""

    _instance = None

    @classmethod
    def instance(cls) -> "FFmpegCLI":
        if cls._instance is None:
            cls._instance = FFmpegCLI()
        return cls._instance

    @cached_property
    def ffmpeg_path(self) -> str:
        bundled = self._bundled_path()
        if _runs(bundled):
            return bundled

        on_path = shutil.which("ffmpeg")
        if _runs(on_path):
            return on_path

        try:
            import imageio_ffmpeg

            wheel_binary = imageio_ffmpeg.get_ffmpeg_exe()
            if _runs(wheel_binary):
                return wheel_binary
        except ImportError:
            pass

        # Nothing runnable found; return the bundled path so callers fail
        # with a clear "cannot execute" error instead of a None crash.
        return bundled

    def _bundled_path(self) -> str:
        system = platform.system()
        if system == "Windows":
            ffmpeg_dir = os.path.join(BASE_DIR, "ffmpeg", "win_x64")
            merge_big_file_if_not_exists(ffmpeg_dir, "ffmpeg.exe")
            binary = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        elif system == "Linux":
            binary = os.path.join(BASE_DIR, "ffmpeg", "linux_x64", "ffmpeg")
        else:
            binary = os.path.join(BASE_DIR, "ffmpeg", "macos", "ffmpeg")
        if os.path.exists(binary):
            os.chmod(binary, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        return binary

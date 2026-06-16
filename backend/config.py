"""Application configuration, translations, and project constants.

The ``Config`` schema, item keys, and defaults are kept byte-for-byte
compatible with existing saved ``config/config.json`` files and with the UI,
which imports these names directly (including the historical ``intefaceTexts``
and ``HARDWARD_ACCELERATION_OPTION`` spellings — do not rename them).
"""

import configparser
import os
from pathlib import Path

from qfluentwidgets import (
    BoolValidator,
    ConfigItem,
    ConfigValidator,
    EnumSerializer,
    OptionsConfigItem,
    OptionsValidator,
    QConfig,
    RangeConfigItem,
    RangeValidator,
    qconfig,
)

from backend.tools.constant import InpaintMode, SubtitleDetectMode

# Project metadata (imported by the UI and the update checker).
VERSION = "1.4.0"
PROJECT_HOME_URL = "https://github.com/sueun-dev/video-text-eraser"
PROJECT_ISSUES_URL = PROJECT_HOME_URL + "/issues"
PROJECT_RELEASES_URL = PROJECT_HOME_URL + "/releases"
PROJECT_UPDATE_URLS = [
    "https://api.github.com/repos/sueun-dev/video-text-eraser/releases/latest",
]

# Master switch for hardware acceleration (read by the settings UI).
HARDWARD_ACCELERATION_OPTION = True


class Config(QConfig):
    """Persistent settings backed by qfluentwidgets' QConfig."""

    # UI language: display name -> interface .ini code.
    intefaceTexts = {
        "简体中文": "ch",
        "繁體中文": "chinese_cht",
        "English": "en",
        "한국어": "ko",
        "日本語": "japan",
        "Tiếng Việt": "vi",
        "Español": "es",
    }
    interface = OptionsConfigItem(
        "Window", "Interface", "ChineseSimplified",
        OptionsValidator(intefaceTexts.values()), restart=True,
    )

    # Window geometry.
    windowX = ConfigItem("Window", "X", None)
    windowY = ConfigItem("Window", "Y", None)
    windowW = ConfigItem("Window", "Width", 1200)
    windowH = ConfigItem("Window", "Height", 1200)

    # All subtitle selection areas in one item, as
    # "ymin,ymax,xmin,xmax;ymin,ymax,xmin,xmax;..." (semicolon-separated).
    subtitleSelectionAreas = ConfigItem(
        "Main", "SubtitleSelectionAreas", "0.88,0.99,0.15,0.85"
    )

    # Inpaint algorithm. STTN_AUTO: fast smart erase, no detection. STTN_DET:
    # detection-driven. LAMA: good for animation/stills. PROPAINTER: heavy VRAM,
    # best on violent motion.
    inpaintMode = OptionsConfigItem(
        "Main", "InpaintMode", InpaintMode.STTN_AUTO,
        OptionsValidator(InpaintMode), EnumSerializer(InpaintMode),
    )
    subtitleDetectMode = OptionsConfigItem(
        "Main", "SubtitleDetectMode", SubtitleDetectMode.PP_OCRv5_SERVER,
        OptionsValidator(SubtitleDetectMode), EnumSerializer(SubtitleDetectMode),
    )

    # A box taller than wide by more than this many pixels is treated as a
    # false text detection and ignored.
    subtitleYXAxisDifferencePixel = RangeConfigItem(
        "Main", "SubtitleYXAxisDifferencePixel", 10, RangeValidator(0, 300)
    )
    # Pixels added around each detected box so glyph edges leave no residue.
    subtitleAreaDeviationPixel = RangeConfigItem(
        "Main", "SubtitleAreaDeviationPixel", 10, RangeValidator(1, 300)
    )
    # X/Y tolerances for deciding two boxes are the same text region.
    subtitleAreaPixelToleranceYPixel = RangeConfigItem(
        "Main", "SubtitleAreaPixelToleranceYPixel", 20, RangeValidator(0, 300)
    )
    subtitleAreaPixelToleranceXPixel = RangeConfigItem(
        "Main", "SubtitleAreaPixelToleranceXPixel", 20, RangeValidator(0, 300)
    )
    # Frames to pad each detected subtitle run backward/forward in time.
    subtitleTimelineBackwardFrameCount = RangeConfigItem(
        "Main", "SubtitleTimelineBackwardFrameCount", 3, RangeValidator(0, 300)
    )
    subtitleTimelineForwardFrameCount = RangeConfigItem(
        "Main", "subtitleTimelineForwardFrameCount", 3, RangeValidator(0, 300)
    )

    # STTN parameters. NeighborStride: spacing of reference frames around the
    # target. ReferenceLength: number of context frames. MaxLoadNum: frames per
    # chunk (must stay >= NeighborStride * ReferenceLength).
    sttnNeighborStride = RangeConfigItem("Sttn", "NeighborStride", 5, RangeValidator(1, 100))
    sttnReferenceLength = RangeConfigItem("Sttn", "ReferenceLength", 10, RangeValidator(1, 100))
    sttnMaxLoadNum = RangeConfigItem("Sttn", "MaxLoadNum", 50, RangeValidator(1, 300))

    def getSttnMaxLoadNum(self) -> int:
        """STTN chunk size; never below stride * reference length."""
        return max(
            self.sttnMaxLoadNum.value,
            self.sttnNeighborStride.value * self.sttnReferenceLength.value,
        )

    # ProPainter frames per pass. Higher = better quality but more VRAM
    # (720p@80 ~25GB, @50 ~19GB; 480p@80 ~8GB, @50 ~7GB).
    propainterMaxLoadNum = RangeConfigItem("ProPainter", "MaxLoadNum", 70, RangeValidator(1, 300))

    hardwareAcceleration = ConfigItem(
        "Main", "HardwareAcceleration", HARDWARD_ACCELERATION_OPTION, BoolValidator()
    )
    checkUpdateOnStartup = ConfigItem("Main", "CheckUpdateOnStartup", True, BoolValidator())
    saveDirectory = ConfigItem("Main", "SaveDirectory", "", ConfigValidator())


CONFIG_FILE = "config/config.json"
config = Config()
qconfig.load(CONFIG_FILE, config)

# Backward compatibility: old SubtitleDetectMode values were Chinese strings.
_detect_mode_value = config.subtitleDetectMode.value
if isinstance(_detect_mode_value, str) and _detect_mode_value in ("快速", "Fast"):
    config.set(config.subtitleDetectMode, SubtitleDetectMode.PP_OCRv5_MOBILE)
elif isinstance(_detect_mode_value, str) and _detect_mode_value in ("精准", "Precise"):
    config.set(config.subtitleDetectMode, SubtitleDetectMode.PP_OCRv5_SERVER)

# Load the UI translation strings for the configured language.
tr = configparser.ConfigParser()
TRANSLATION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "interface", f"{config.interface.value}.ini"
)
tr.read(TRANSLATION_FILE, encoding="utf-8")

# Project base directory.
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

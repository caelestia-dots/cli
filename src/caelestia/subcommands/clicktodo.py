# clicktodo.py — fast, subtle, theme-aware overlay (refactor)
# - Smoother scan→idle→interaction transitions with cross-fades
# - Zero white-flash by painting before show + translucent window
# - No per-frame object churn (gradients, pens, paths cached)
# - Minimal allocations in paintEvent; only dirty state triggers update
# - Tight integration with Caelestia scheme

import math
import os
import subprocess
import tempfile
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from caelestia.utils.notify import notify
from caelestia.utils.scheme import get_scheme
from caelestia.ocr_client import StreamNotSupportedError, stream_ocr as stream_ocr_messages

# --------------------------- Core model types ---------------------------

@dataclass
class RecognizedRegion:
    polygon: List[Tuple[float, float]]
    bbox: Tuple[float, float, float, float]
    text: str
    confidence: float


# ------------------------------ Debugging ------------------------------

def _is_debug_enabled(args: Namespace | None = None) -> bool:
    env_value = os.getenv("CAELESTIA_DEBUG", "")
    env_enabled = env_value.lower() in {"1", "true", "yes", "on"}
    arg_enabled = bool(getattr(args, "debug", False)) if args is not None else False
    return arg_enabled or env_enabled


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[clicktodo] {message}", flush=True)


# -------------------------- OCR service helpers ------------------------

def ensure_ocr_service_ready(debug: bool = False):
    """Ensure the OCR daemon is up before capturing a screenshot."""
    try:
        from caelestia.ocr_client import get_ocr_client
    except ImportError:
        raise ImportError("OCR client not available. Please ensure the package is properly installed.")

    start = time.perf_counter()
    client = get_ocr_client()
    debug_log(debug, "Ensuring OCR daemon is ready")

    ensure_daemon = getattr(client, "_ensure_daemon", None)
    if ensure_daemon is None or not callable(ensure_daemon):
        raise RuntimeError("OCR client missing daemon bootstrap helper")

    if not ensure_daemon():
        raise RuntimeError("Could not start OCR daemon. Please install dependencies: pip install rapidocr-onnxruntime")

    elapsed_ms = (time.perf_counter() - start) * 1000
    stats = client.get_stats() if hasattr(client, "get_stats") else None

    if stats:
        warmed = stats.get("warmed", False)
        avg_ms = stats.get("avg_time", 0.0) * 1000
        debug_log(
            debug,
            "OCR daemon ready (requests=%s avg=%.2fms warmed=%s warmup=%.1fms)"
            % (stats.get("requests", 0), avg_ms, warmed, elapsed_ms),
        )
    else:
        debug_log(debug, f"OCR daemon ready (warmup took {elapsed_ms:.1f}ms)")

    return client


def warm_up_ocr(client, fast: bool, debug: bool) -> None:
    """Explicitly warm up the OCR daemon to keep models hot."""
    stats = None
    try:
        stats = client.get_stats()
    except Exception as exc:
        debug_log(debug, f"Failed to fetch stats before warm-up: {exc}")

    if stats and stats.get("warmed") and stats.get("requests", 0) > 0:
        debug_log(debug, "Skipping warm-up; daemon already hot")
        return

    try:
        debug_log(debug, f"Running warm-up inference (fast={fast})")
        response = client.warm_up(fast=fast)
        warm_ms = response.get("timing", {}).get("warm")
        if warm_ms is not None:
            debug_log(debug, f"Warm-up completed in {warm_ms:.1f}ms")
    except Exception as exc:
        debug_log(debug, f"Warm-up failed: {exc}")


# ----------------------------- CLI entrypoint --------------------------

class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        debug = _is_debug_enabled(self.args)
        fast_mode = getattr(self.args, "fast", False)
        live_mode = getattr(self.args, "live", False)

        try:
            debug_log(debug, f"Fast mode {'enabled' if fast_mode else 'disabled'}")
            client = ensure_ocr_service_ready(debug=debug)
            warm_up_ocr(client, fast=fast_mode, debug=debug)

            # Capture fullscreen
            image_path = capture_fullscreen(debug=debug)
            debug_log(debug, f"Screenshot captured to {image_path}")

            selected_text, region_count, cancelled = launch_overlay(
                image_path,
                fast=fast_mode,
                debug=debug,
                live=live_mode,
            )

            if cancelled and region_count == 0 and not selected_text:
                debug_log(debug, "Overlay cancelled before OCR completed")
                return

            if cancelled and region_count > 0:
                debug_log(debug, "Overlay cancelled by user")
                return

            if region_count == 0:
                debug_log(debug, "No text regions detected; notifying user")
                notify("OCR Click-to-Copy", "No text detected in screenshot")
                return

            if selected_text:
                notify("OCR Click-to-Copy", f"Copied: {selected_text[:50]}{'...' if len(selected_text) > 50 else ''}")
                debug_log(debug, f"Copied text: {selected_text}")
            else:
                debug_log(debug, "Overlay closed without selection")

        except ImportError as e:
            # Show user-friendly message for missing dependencies
            debug_log(debug, f"Import error: {e}")
            notify("OCR Click-to-Copy", str(e))
            print(f"Error: {e}", file=__import__("sys").stderr)
        except Exception as e:
            notify("OCR Click-to-Copy", f"Error: {str(e)}")
            debug_log(debug, f"Unhandled error: {e}")
            print(f"Error: {e}", file=__import__("sys").stderr)


# ------------------------------- Capture --------------------------------

def capture_fullscreen(debug: bool = False) -> str:
    # JPEG @ q=90 is generally fastest with grim while keeping size small
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()

    cmd = ["grim", "-t", "jpeg", "-q", "90", tmp_path]
    debug_log(debug, f"Capturing fullscreen via: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True)
    t1 = (time.perf_counter() - t0) * 1000
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    debug_log(debug, f"Screenshot done in {t1:.1f} ms")
    return tmp_path


# ------------------------------ OCR runner ------------------------------

def run_ocr_on_image(image_path: str, fast: bool = False, debug: bool = False) -> List[RecognizedRegion]:
    """
    Run OCR on an image to detect text and bounding boxes using the OCR daemon.
    """
    try:
        from caelestia.ocr_client import ocr_full
    except ImportError:
        raise ImportError("OCR client not available. Please ensure the package is properly installed.")

    try:
        debug_log(debug, f"Requesting OCR from daemon (fast={fast})")
        start = time.perf_counter()
        boxes, texts, scores = ocr_full(image_path, fast=fast)
        duration_ms = (time.perf_counter() - start) * 1000
        debug_log(debug, f"OCR finished in {duration_ms:.1f}ms")

        regions: List[RecognizedRegion] = []
        for box, text, confidence in zip(boxes, texts, scores):
            polygon = [(float(p[0]), float(p[1])) for p in box]
            xs = [p[0] for p in polygon] or [0.0]
            ys = [p[1] for p in polygon] or [0.0]
            x0, y0 = float(min(xs)), float(min(ys))
            x1, y1 = float(max(xs)), float(max(ys))
            regions.append(
                RecognizedRegion(
                    polygon=polygon,
                    bbox=(x0, y0, x1, y1),
                    text=text,
                    confidence=float(confidence),
                )
            )
        return regions

    except Exception as e:
        # Provide helpful error message if daemon can't start
        raise RuntimeError(
            f"OCR processing failed: {e}\n\n"
            "The OCR daemon requires rapidocr-onnxruntime.\n"
            "Install with: pip install rapidocr-onnxruntime\n"
            "Or: pip install caelestia[ocr]"
        )


# ------------------------------ Overlay UI ------------------------------

def launch_overlay(
    image_path: str,
    fast: bool = False,
    debug: bool = False,
    live: bool = False,
) -> tuple[str | None, int, bool]:
    """Combined scanning animation + interaction UI with subtle transitions."""
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow
        from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer, QElapsedTimer, QThread, pyqtSignal
        from PyQt6.QtGui import (
            QPainter, QColor, QPen, QFont, QPainterPath, QPixmap,
            QPolygonF, QFontMetrics
        )
        import sys
    except ImportError:
        raise ImportError(
            "PyQt6 is not installed. Install it with: pip install PyQt6\n"
            "Or install with the 'ocr' extra: pip install caelestia[ocr]"
        )

    # ---------------------------- Theme plumbing ----------------------------

    @dataclass(slots=True)
    class OverlayTheme:
        # Backdrop + scan
        backdrop_tint: QColor
        backdrop_idle_alpha: float
        backdrop_scan_alpha: float
        scan_far: QColor
        scan_mid: QColor
        scan_peak: QColor
        scan_glow: QColor
        # Idle background shimmer
        idle_start: QColor
        idle_mid: QColor
        idle_end: QColor
        idle_glow: QColor
        # Regions
        region_fill: QColor
        region_hover_fill: QColor
        region_selection_fill: QColor
        region_border: QColor
        region_hover_border: QColor
        region_selection_border: QColor
        selection_glow: QColor
        # Characters
        char_hover_fill: QColor
        char_selection_fill: QColor
        char_selection_outline: QColor
        # Help bubble
        help_background: QColor
        help_border: QColor
        help_text: QColor
        # Misc
        border_width: float

    # -------------------------- Worker (stream/full) ------------------------

    class OcrWorker(QThread):
        detections_ready = pyqtSignal(object)
        partial_ready = pyqtSignal(object)
        done = pyqtSignal(object)
        error = pyqtSignal(str)

        def __init__(self, image_path: str, fast: bool, debug: bool, live: bool) -> None:
            super().__init__()
            self.image_path = image_path
            self.fast = fast
            self.debug = debug
            self.live = live

        def run(self) -> None:  # pragma: no cover - compositor required
            if self.live:
                if self._run_stream():
                    return
            self._run_full()

        def _run_stream(self) -> bool:
            try:
                for message in stream_ocr_messages(self.image_path, fast=self.fast):
                    msg_type = message.get("type")
                    if msg_type == "det":
                        self.detections_ready.emit(message.get("boxes") or [])
                    elif msg_type == "update":
                        region = self._region_from_message(message)
                        if region:
                            self.partial_ready.emit(region)
                    elif msg_type == "error":
                        self.error.emit(message.get("message", "Streaming error"))
                    elif msg_type == "done":
                        self.done.emit(message)
                return True
            except StreamNotSupportedError:
                debug_log(self.debug, "Streaming not supported by daemon, falling back")
                return False
            except Exception as exc:  # propagate to UI
                self.error.emit(str(exc))
                return True

        def _run_full(self) -> None:
            try:
                regions = run_ocr_on_image(self.image_path, fast=self.fast, debug=self.debug)
            except Exception as exc:
                self.error.emit(str(exc))
                return

            boxes = [[[float(x), float(y)] for (x, y) in r.polygon] for r in regions]
            if boxes:
                self.detections_ready.emit(boxes)
            for r in regions:
                self.partial_ready.emit(r)
            self.done.emit({"emitted": len(regions), "detected": len(boxes), "fallback": True})

        def _region_from_message(self, message: Dict) -> Optional[RecognizedRegion]:
            raw_box = message.get("box") or []
            if not raw_box:
                return None
            polygon = [(float(p[0]), float(p[1])) for p in raw_box]
            bbox = message.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                bbox_tuple = tuple(float(v) for v in bbox)
            else:
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                bbox_tuple = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
            return RecognizedRegion(
                polygon=polygon,
                bbox=bbox_tuple,
                text=message.get("text", ""),
                confidence=float(message.get("conf", 0.0)),
            )

    # ------------------------------ Main window -----------------------------

    class OverlayWindow(QMainWindow):
        # Inner layout wrapper for regions (precompute once)
        class TextRegionLayout:
            def __init__(self, region: RecognizedRegion, scale_factor: float):

                self.region = region
                self.text = region.text or ""
                self.confidence = region.confidence
                self.polygon = QPolygonF([QPointF(x / scale_factor, y / scale_factor) for x, y in region.polygon])

                if self.polygon.isEmpty():
                    x0, y0, x1, y1 = region.bbox
                    self.polygon = QPolygonF(
                        [QPointF(x0 / scale_factor, y0 / scale_factor),
                         QPointF(x1 / scale_factor, y0 / scale_factor),
                         QPointF(x1 / scale_factor, y1 / scale_factor),
                         QPointF(x0 / scale_factor, y1 / scale_factor)]
                    )

                self.path = QPainterPath()
                self.path.addPolygon(self.polygon)
                self.bounding_rect: QRectF = self.path.boundingRect()
                self.orientation = self._detect_orientation()
                self.font = QFont()
                self.char_rects: List["QRectF"] = []
                self.display_rect: "QRectF" = self.bounding_rect
                self.display_path: QPainterPath = self.path
                self._layout_characters()

            def _edge_angle(self) -> float | None:
                if len(self.region.polygon) < 2:
                    return None
                (x0, y0), (x1, y1) = self.region.polygon[0], self.region.polygon[1]
                vx, vy = x1 - x0, y1 - y0
                if vx == 0 and vy == 0:
                    return None
                angle = abs(math.degrees(math.atan2(vy, vx)))
                return 180 - angle if angle > 90 else angle

            def _detect_orientation(self) -> str:
                w, h = self.bounding_rect.width(), self.bounding_rect.height()
                if w <= 0 or h <= 0 or len(self.text.strip()) <= 1:
                    return "horizontal"
                edge = self._edge_angle()
                if edge is not None and edge > 45:
                    return "vertical"
                if h > w * 1.6:
                    return "vertical"
                return "horizontal"

            def _layout_characters(self) -> None:
                available = self.bounding_rect
                if available.width() <= 0 or available.height() <= 0:
                    self.char_rects = []
                    self.display_rect = available
                    return

                # Find font size to fill region while avoiding reflow in paint
                min_font, max_font = 8, 72
                if self.orientation == "horizontal":
                    start = int(max(min(available.height(), max_font), min_font))
                    font_size = self._fit_horizontal_font(start, min_font, available.width())
                else:
                    start = int(max(min(available.width(), max_font), min_font))
                    font_size = self._fit_vertical_font(start, min_font, available.height())

                self.font.setPixelSize(font_size)
                metrics = QFontMetrics(self.font)

                if self.orientation == "horizontal":
                    self.char_rects = self._generate_horizontal_rects(metrics, available)
                else:
                    self.char_rects = self._generate_vertical_rects(metrics, available)
                self.display_rect = available
                self.display_path = self._build_display_path()
                
                # Detect word boundaries for smart selection
                self._detect_word_boundaries()

            def _fit_horizontal_font(self, start: int, min_font: int, max_width: float) -> int:
                test = QFont()
                size = max(start, min_font)
                while size > min_font:
                    test.setPixelSize(size)
                    total = sum(max(QFontMetrics(test).horizontalAdvance(ch), 1) for ch in self.text)
                    if total <= max_width or len(self.text) <= 1:
                        break
                    size -= 1
                return max(size, min_font)

            def _fit_vertical_font(self, start: int, min_font: int, max_height: float) -> int:
                test = QFont()
                size = max(start, min_font)
                while size > min_font:
                    test.setPixelSize(size)
                    total = QFontMetrics(test).height() * max(len(self.text), 1)
                    if total <= max_height or len(self.text) <= 1:
                        break
                    size -= 1
                return max(size, min_font)

            def _generate_horizontal_rects(self, metrics: "QFontMetrics", available) -> List["QRectF"]:
                if not self.text:
                    return []
                width = available.width()
                if width <= 0:
                    return []
                raw = [max(float(metrics.horizontalAdvance(ch)), 1.0) for ch in self.text]
                total_width = sum(raw)
                scale = (width / total_width) if total_width > 0 else 1.0
                rects: List[QRectF] = []
                x = available.left()
                right = available.left() + width
                
                for idx, w in enumerate(raw):
                    scaled_w = w * scale
                    # Ensure last character fills to the right edge
                    if idx == len(raw) - 1:
                        scaled_w = right - x
                    # Ensure minimum width for visibility
                    scaled_w = max(scaled_w, 1.0)
                    rects.append(QRectF(x, available.top(), scaled_w, available.height()))
                    x += scaled_w
                return rects

            def _generate_vertical_rects(self, metrics: "QFontMetrics", available) -> List["QRectF"]:
                if not self.text:
                    return []
                height = available.height()
                if height <= 0:
                    return []
                raw_h = max(float(metrics.height()), 1.0)
                total_height = raw_h * len(self.text)
                scale = (height / total_height) if total_height > 0 else 1.0
                rects: List[QRectF] = []
                y = available.top()
                bottom = available.top() + height
                w = available.width()
                
                for idx, _ in enumerate(self.text):
                    scaled_h = raw_h * scale
                    # Ensure last character fills to the bottom edge
                    if idx == len(self.text) - 1:
                        scaled_h = bottom - y
                    # Ensure minimum height for visibility
                    scaled_h = max(scaled_h, 1.0)
                    rects.append(QRectF(available.left(), y, w, scaled_h))
                    y += scaled_h
                return rects

            def _detect_word_boundaries(self) -> None:
                """Detect word boundaries in the text for smart selection."""
                import re
                self.word_boundaries = []  # List of (start, end) tuples for each word
                
                if not self.text:
                    return
                
                # Find all word-like sequences (alphanumeric, including unicode)
                # This regex matches words, numbers, and preserves common punctuation as separate tokens
                pattern = r'\w+|[^\w\s]'
                for match in re.finditer(pattern, self.text, re.UNICODE):
                    self.word_boundaries.append((match.start(), match.end()))
            
            def get_word_at(self, char_idx: int) -> tuple[int, int] | None:
                """Get the word boundaries containing the given character index."""
                if not hasattr(self, 'word_boundaries'):
                    return None
                
                for start, end in self.word_boundaries:
                    if start <= char_idx < end:
                        return (start, end)
                return None
            
            def snap_to_word_boundaries(self, start_idx: int, end_idx: int) -> tuple[int, int]:
                """
                Intelligently snap selection to word boundaries.
                If selection spans multiple characters within a word, expand to full word.
                If it crosses word boundaries, respect the user's selection.
                """
                if not hasattr(self, 'word_boundaries') or start_idx == end_idx:
                    return (start_idx, end_idx)
                
                # Normalize order
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                
                # Find words at start and end
                start_word = self.get_word_at(start_idx)
                end_word = self.get_word_at(max(0, end_idx - 1))  # end_idx is exclusive
                
                # If both are in the same word and selection covers >2 chars, expand to whole word
                if start_word and end_word and start_word == end_word:
                    if (end_idx - start_idx) >= 2:
                        return start_word
                    else:
                        return (start_idx, end_idx)
                
                # If spanning multiple words, snap start to word beginning and end to word end
                if start_word and end_word:
                    return (start_word[0], end_word[1])
                
                # Single word at start
                if start_word:
                    return (start_word[0], end_idx)
                
                # Single word at end
                if end_word:
                    return (start_idx, end_word[1])
                
                return (start_idx, end_idx)

            def index_at(self, pos) -> int:
                """Get character index at position. Returns index where cursor should be placed."""
                if not self.text or not self.char_rects:
                    return 0
                
                if self.orientation == "horizontal":
                    click_x = pos.x()
                    
                    # Check if before first character
                    if click_x < self.char_rects[0].left():
                        return 0
                    
                    # Check if after last character
                    if click_x >= self.char_rects[-1].right():
                        return len(self.text)
                    
                    # Find the character rect containing the click
                    for i, r in enumerate(self.char_rects):
                        if click_x >= r.left() and click_x < r.right():
                            # Determine if click is in left or right half
                            char_center = r.left() + r.width() / 2.0
                            if click_x < char_center:
                                return i  # Left half - cursor before this character
                            else:
                                return i + 1  # Right half - cursor after this character
                    
                    return len(self.text)
                else:
                    click_y = pos.y()
                    
                    # Check if before first character
                    if click_y < self.char_rects[0].top():
                        return 0
                    
                    # Check if after last character
                    if click_y >= self.char_rects[-1].bottom():
                        return len(self.text)
                    
                    # Find the character rect containing the click
                    for i, r in enumerate(self.char_rects):
                        if click_y >= r.top() and click_y < r.bottom():
                            # Determine if click is in top or bottom half
                            char_center = r.top() + r.height() / 2.0
                            if click_y < char_center:
                                return i  # Top half - cursor before this character
                            else:
                                return i + 1  # Bottom half - cursor after this character
                    
                    return len(self.text)

            def _build_display_path(self) -> QPainterPath:
                rect = self.display_rect
                if rect.width() <= 0 or rect.height() <= 0:
                    return self.path
                radius = max(3.5, min((min(rect.width(), rect.height()) * 0.18), min(rect.width(), rect.height()) / 2))
                rounded = QPainterPath()
                rounded.addRoundedRect(rect, radius, radius)
                hit = rounded.intersected(self.path)
                return hit if not hit.isEmpty() else rounded

        def __init__(self, bg_image_path: str, fast: bool, debug: bool, live: bool):
            super().__init__()
            self.bg_image_path = bg_image_path
            self.fast = fast
            self.debug = debug
            self.live = live
            debug_log(self.debug, f"Live mode {'enabled' if self.live else 'disabled'}")

            # Selection state
            self.hovered_index: int | None = None
            self.selected_text: str | None = None
            self.selection_start: Tuple[int, int] | None = None
            self.selection_end: Tuple[int, int] | None = None
            self.is_selecting = False
            self.was_cancelled = False

            # Regions
            self.total_regions: int = 0
            self.expected_regions: int = 0
            self.regions: List["OverlayWindow.TextRegionLayout"] = []
            self.region_lookup: Dict[str, int] = {}
            self.stream_completed: bool = False

            # Phases + timing
            self.phase: str = "scanning"  # scanning -> waiting_results -> interaction
            self.scan_progress: float = 0.0
            self.scan_complete: bool = False
            self.scan_beam_active: bool = True
            self.scan_duration_ms = 800
            self.scan_hold_ms = 100

            # Crossfade (prevents abrupt jump on finish)
            self.phase_fade: float = 0.0  # 0..1
            self.phase_fade_active: bool = False

            # Simple fade timer for phase transitions only
            self.fade_timer = QTimer(self)
            self.fade_timer.setInterval(50)
            self.fade_timer.timeout.connect(self._update_phase_fade)
            self.fade_timer.setSingleShot(False)

            self.ocr_error: str | None = None

            # Scan timer for phase fade only
            self.scan_timer = QTimer(self)
            self.scan_timer.setInterval(50)
            self.scan_timer.timeout.connect(self._update_scan)
            self.scan_elapsed = QElapsedTimer()

            # Graceful exit timer
            self.cleanup_timer = QTimer(self)
            self.cleanup_timer.setSingleShot(True)
            self.cleanup_timer.timeout.connect(QApplication.quit)

            # Background screenshot
            self.bg_pixmap = QPixmap(bg_image_path)
            if self.bg_pixmap.isNull():
                self.ocr_error = "Failed to load screenshot for overlay"
                self.cleanup_timer.start(0)
                return

            # Determine scale
            try:
                result = subprocess.run(["hyprctl", "monitors", "-j"], capture_output=True, text=True, check=True)
                import json
                monitors = json.loads(result.stdout)
                self.scale_factor = monitors[0].get("scale", 1.0) if monitors else 1.0
            except Exception:
                self.scale_factor = 1.0

            self.screen_width = int(self.bg_pixmap.width() / max(self.scale_factor, 1e-3))
            self.screen_height = int(self.bg_pixmap.height() / max(self.scale_factor, 1e-3))

            # Theme + caches
            self.theme = self._create_theme()
            self._cache = {}  # lazy cache for gradients/pens

            # Window flags to avoid white flash
            self.setWindowTitle("OCR Click-to-Copy")
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            # Crucial bits: no system background + translucent + prepaint before show
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            self.setMouseTracking(True)
            self.setGeometry(0, 0, self.screen_width, self.screen_height)
            self.setFixedSize(self.screen_width, self.screen_height)

            # Precompute one paint before showing (prevents flash)
            self._first_frame_ready = False
            QTimer.singleShot(0, self._paint_once_then_show)

            # OCR worker
            self.ocr_worker = OcrWorker(bg_image_path, fast, debug, live)
            self.ocr_worker.detections_ready.connect(self._on_detections_ready)
            self.ocr_worker.partial_ready.connect(self._on_region_ready)
            self.ocr_worker.done.connect(self._on_stream_done)
            self.ocr_worker.error.connect(self._on_ocr_error)
            self.ocr_worker.start()

            self._start_scan()

        # -------------------------- Setup & theme --------------------------

        def _paint_once_then_show(self):
            # Trigger one paint offscreen then show; avoids white flash
            self._first_frame_ready = True
            self.showFullScreen()

        def _create_theme(self) -> OverlayTheme:
            border_width = max(1.4, 2.0 / max(self.scale_factor, 0.5))
            try:
                scheme = get_scheme()
                colours = dict(getattr(scheme, "colours", {}) or {})
            except Exception as exc:
                colours = {}
                debug_log(self.debug, f"Falling back to default overlay palette: {exc}")

            def pick(name: str, fallback: str, alt: str | None = None) -> "QColor":
                raw = colours.get(name) or (colours.get(alt) if alt else None) or fallback
                if not raw.startswith("#"):
                    raw = f"#{raw}"
                return QColor(raw)

            def with_alpha(color: "QColor", alpha: float) -> "QColor":
                c = QColor(color)
                c.setAlphaF(max(0.0, min(alpha, 1.0)))
                return c

            def lighten(color: "QColor", factor: int) -> "QColor":
                return QColor(color).lighter(factor)

            def mix(a: "QColor", b: "QColor", r: float) -> "QColor":
                r = max(0.0, min(r, 1.0))
                inv = 1.0 - r
                return QColor(int(a.red()*inv + b.red()*r),
                              int(a.green()*inv + b.green()*r),
                              int(a.blue()*inv + b.blue()*r))

            primary = pick("primary", "4d9dff")
            secondary = pick("secondary", "71c1ff")
            surface = pick("surface", "0d1524")
            on_surface = pick("onSurface", "e5ecf6")

            def alpha(c, a): 
                n = QColor(c)
                n.setAlphaF(a)
                return n

            # Subtle but visible palette
            backdrop_tint = alpha(surface, 0.80)
            backdrop_idle_alpha = 0.10
            backdrop_scan_alpha = 0.10

            scan_peak = alpha(primary, 0.40)
            scan_mid = alpha(secondary, 0.30)
            scan_far = alpha(surface, 0.20)
            scan_glow = alpha(secondary, 0.20)

            idle_start = alpha(surface, 0.10)
            idle_mid = alpha(primary, 0.08)
            idle_end = alpha(secondary, 0.07)
            idle_glow = alpha(primary, 0.07)

            region_fill = alpha(primary, 0.08)
            region_hover_fill = alpha(secondary, 0.16)
            region_selection_fill = alpha(primary, 0.35)  # Much more visible

            region_border = alpha(primary, 0.40)
            region_hover_border = alpha(secondary, 0.55)
            region_selection_border = alpha(primary, 0.75)  # Stronger border
            selection_glow = alpha(secondary, 0.35)  # Stronger glow

            char_hover_fill = alpha(primary, 0.08)
            char_selection_fill = alpha(primary, 0.12)
            char_selection_outline = alpha(secondary, 0.6)

            help_background = alpha(surface, 0.92)
            help_border = alpha(primary, 0.35)
            help_text = alpha(on_surface, 0.9)

            border_width = 1.0
        

            return OverlayTheme(
                backdrop_tint=backdrop_tint,
                backdrop_idle_alpha=backdrop_idle_alpha,
                backdrop_scan_alpha=backdrop_scan_alpha,
                scan_far=scan_far,
                scan_mid=scan_mid,
                scan_peak=scan_peak,
                scan_glow=scan_glow,
                idle_start=idle_start,
                idle_mid=idle_mid,
                idle_end=idle_end,
                idle_glow=idle_glow,
                region_fill=region_fill,
                region_hover_fill=region_hover_fill,
                region_selection_fill=region_selection_fill,
                region_border=region_border,
                region_hover_border=region_hover_border,
                region_selection_border=region_selection_border,
                selection_glow=selection_glow,
                char_hover_fill=char_hover_fill,
                char_selection_fill=char_selection_fill,
                char_selection_outline=char_selection_outline,
                help_background=help_background,
                help_border=help_border,
                help_text=help_text,
                border_width=border_width,
            )

        # ----------------------------- Lifecycle -----------------------------

        def _start_scan(self) -> None:
            self.scan_elapsed.start()
            self.scan_timer.start()
            debug_log(self.debug, "Scan animation started")

        def _update_scan(self) -> None:
            if self.scan_complete:
                return
            elapsed = self.scan_elapsed.elapsed()
            self.scan_progress = 1.0 if self.scan_duration_ms <= 0 else min(1.0, elapsed / self.scan_duration_ms)
            if self.scan_progress >= 1.0 and not self.scan_complete:
                self.scan_complete = True
                self.scan_beam_active = False
                self.scan_timer.stop()
                debug_log(self.debug, "Scan animation finished")
                QTimer.singleShot(self.scan_hold_ms, self._after_scan_hold)
            self.update()

        def _update_phase_fade(self) -> None:
            if self.phase_fade_active and self.phase_fade < 1.0:
                self.phase_fade = min(1.0, self.phase_fade + 0.15)
                self.update()
                if self.phase_fade >= 1.0:
                    self.fade_timer.stop()
            else:
                self.fade_timer.stop()

        def _after_scan_hold(self) -> None:
            # Begin crossfade to next phase instead of abrupt switch
            self.phase_fade = 0.0
            self.phase_fade_active = True
            self.phase = "interaction" if self.total_regions > 0 else "waiting_results"
            self.fade_timer.start()
            self.update()

        def _finish_scan_animation(self) -> None:
            if not self.scan_complete:
                self.scan_complete = True
                self.scan_beam_active = False
                self.scan_timer.stop()

        # ------------------------------- OCR IO ------------------------------

        def _region_key(self, polygon: List[Tuple[float, float]]) -> str:
            return "|".join(f"{int(round(x))}:{int(round(y))}" for x, y in polygon)

        def _on_detections_ready(self, boxes: List[List[List[float]]]) -> None:
            count = len(boxes or [])
            self.expected_regions = max(self.expected_regions, count)
            if count:
                debug_log(self.debug, f"Daemon detected {count} candidate regions")
            self.update()

        def _on_region_ready(self, region: RecognizedRegion) -> None:
            self._finish_scan_animation()
            layout = self.TextRegionLayout(region, self.scale_factor)
            key = self._region_key(region.polygon)

            if key in self.region_lookup:
                self.regions[self.region_lookup[key]] = layout
            else:
                self.region_lookup[key] = len(self.regions)
                self.regions.append(layout)

            self.total_regions = len(self.regions)
            if self.phase != "interaction":
                # Smoothly crossfade in if we were waiting
                self.phase = "interaction"
                self.phase_fade = 0.0
                self.phase_fade_active = True
                self.fade_timer.start()
            self.update()

        def _on_stream_done(self, summary: Dict | None) -> None:
            self.stream_completed = True
            self._finish_scan_animation()

            if isinstance(summary, dict):
                detected = summary.get("detected")
                if isinstance(detected, int):
                    self.expected_regions = max(self.expected_regions, detected)
                if summary.get("fallback"):
                    debug_log(self.debug, "Stream fallback completed with full OCR")

            if self.total_regions == 0:
                debug_log(self.debug, "OCR completed without detecting text")
                self.phase = "waiting_results"
                # Gentle fade out to exit
                self.phase_fade = 0.0
                self.phase_fade_active = True
                self.cleanup_timer.start(240)
            else:
                self.phase = "interaction"

            self.update()

        def _on_ocr_error(self, message: str) -> None:
            self.ocr_error = message
            debug_log(self.debug, f"OCR error: {message}")
            self.cleanup_timer.start(0)

        # ------------------------------- Paint -------------------------------

        def _g(self, key: str, builder):
            """Small cache to avoid rebuilding gradients/pens each frame."""
            val = self._cache.get(key)
            if val is None:
                val = builder()
                self._cache[key] = val
            return val

        def _paint_backdrop(self, painter: "QPainter") -> None:
            """Draw minimal, static backdrop - just a subtle tint."""
            # Subtle but visible semi-transparent tint
            base = QColor(self.theme.backdrop_tint)
            base.setAlphaF(self.theme.backdrop_idle_alpha * 1.8)  # More visible
            painter.fillRect(self.rect(), base)



        def _paint_regions(self, painter: "QPainter", fade: float) -> None:
            if not self.regions:
                return

            painter.setPen(Qt.PenStyle.NoPen)

            for idx, layout in enumerate(self.regions):
                is_hovered = self.hovered_index == idx
                
                # Determine character-level selection range for this region
                char_start = -1
                char_end = -1
                if self.selection_start is not None and self.selection_end is not None:
                    sb, sc = self.selection_start
                    eb, ec = self.selection_end
                    if sb <= idx <= eb:
                        if idx == sb and idx == eb:
                            # Selection within single region - apply smart word snapping
                            raw_start = min(sc, ec)
                            raw_end = max(sc, ec)
                            char_start, char_end = layout.snap_to_word_boundaries(raw_start, raw_end)
                        elif idx == sb:
                            # Start of multi-region selection
                            char_start = sc
                            char_end = len(layout.text)
                            # Snap start to word boundary
                            word = layout.get_word_at(char_start)
                            if word:
                                char_start = word[0]
                        elif idx == eb:
                            # End of multi-region selection
                            char_start = 0
                            char_end = ec
                            # Snap end to word boundary
                            if ec > 0:
                                word = layout.get_word_at(ec - 1)
                                if word:
                                    char_end = word[1]
                        else:
                            # Middle of multi-region selection
                            char_start = 0
                            char_end = len(layout.text)
                
                has_selection = char_start >= 0 and char_end >= 0
                
                # Choose base color
                if has_selection:
                    base_fill = QColor(self.theme.region_selection_fill)
                    glow = QColor(self.theme.selection_glow)
                    glow_layers = 4
                elif is_hovered:
                    base_fill = QColor(self.theme.region_hover_fill)
                    glow = QColor(self.theme.region_hover_border)
                    glow_layers = 2
                else:
                    base_fill = QColor(self.theme.region_fill)
                    glow = QColor(self.theme.region_border)
                    glow_layers = 1
                
                # Draw subtle glow layers for whole region
                if glow_layers > 0:
                    painter.save()
                    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
                    for layer in range(glow_layers):
                        expand = (layer + 1) * 2.5
                        base_alpha = 0.10 if has_selection else 0.06
                        glow_alpha = (base_alpha * (glow_layers - layer) / glow_layers) * fade
                        glow_color = QColor(glow)
                        glow_color.setAlphaF(glow_alpha)
                        
                        glow_rect = layout.display_rect.adjusted(-expand, -expand, expand, expand)
                        radius = max(4, min((min(glow_rect.width(), glow_rect.height()) * 0.18), 
                                           min(glow_rect.width(), glow_rect.height()) / 2))
                        glow_path = QPainterPath()
                        glow_path.addRoundedRect(glow_rect, radius, radius)
                        painter.fillPath(glow_path, glow_color)
                    painter.restore()
                
                # Draw base fill for whole region
                base_fill.setAlphaF(base_fill.alphaF() * fade * 0.6)  # Lighter base
                painter.fillPath(layout.display_path, base_fill)
                
                # Draw character-level selection highlight
                if has_selection and layout.char_rects:
                    painter.save()
                    sel_fill = QColor(self.theme.char_selection_fill)
                    sel_fill.setAlphaF(sel_fill.alphaF() * fade * 2.5)  # Very visible
                    
                    # Debug: log selection range with word detection info
                    if self.debug:
                        selected_text = layout.text[char_start:char_end]
                        word_info = ""
                        if hasattr(layout, 'word_boundaries'):
                            word_info = f" [words: {layout.word_boundaries}]"
                        debug_log(self.debug, f"Region {idx}: selecting chars [{char_start}:{char_end}] = '{selected_text}'{word_info}")
                    
                    # Highlight each selected character
                    for char_idx in range(char_start, min(char_end, len(layout.char_rects))):
                        if 0 <= char_idx < len(layout.char_rects):
                            char_rect = layout.char_rects[char_idx]
                            # Create rounded rect path for each character
                            char_path = QPainterPath()
                            small_radius = min(3.0, min(char_rect.width(), char_rect.height()) * 0.2)
                            char_path.addRoundedRect(char_rect, small_radius, small_radius)
                            painter.fillPath(char_path, sel_fill)
                    painter.restore()
                
                # Draw text in debug mode
                if self.debug and layout.text:
                    painter.save()
                    painter.setPen(self.theme.help_text)
                    painter.setFont(layout.font)
                    
                    if layout.orientation == "horizontal":
                        painter.drawText(layout.display_rect, Qt.AlignmentFlag.AlignCenter, layout.text)
                    else:
                        # Vertical text - rotate and draw
                        painter.translate(layout.display_rect.center())
                        painter.rotate(90)
                        temp_rect = QRectF(-layout.display_rect.height()/2, -layout.display_rect.width()/2,
                                          layout.display_rect.height(), layout.display_rect.width())
                        painter.drawText(temp_rect, Qt.AlignmentFlag.AlignCenter, layout.text)
                    
                    painter.restore()

        def paintEvent(self, _ev) -> None:
            if not self._first_frame_ready:
                return

            painter = QPainter(self)
            # Minimal antialiasing - only for paths, not pixmaps
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            # Draw bg screen first - exact dimensions, no shifting
            # Use source rect to ensure proper 1:1 mapping
            painter.drawPixmap(
                0, 0, self.screen_width, self.screen_height,
                self.bg_pixmap,
                0, 0, self.bg_pixmap.width(), self.bg_pixmap.height()
            )

            # Backdrop + scan/idle layers
            self._paint_backdrop(painter)

            # Regions with crossfade to avoid abruptness on phase transitions
            fade = 1.0 if not self.phase_fade_active else self.phase_fade
            self._paint_regions(painter, fade)

            # (Optional) help chip when waiting / no results
            if self.phase == "waiting_results":
                tip = "No text detected"
                rect = QRectF(self.screen_width*0.5-110, self.screen_height*0.87-18, 220, 36)
                radius = 10.0
                
                # Simple bubble
                painter.setPen(Qt.PenStyle.NoPen)
                path = QPainterPath()
                path.addRoundedRect(rect, radius, radius)
                painter.fillPath(path, self.theme.help_background)
                
                # Border
                border_pen = QPen(self.theme.help_border, 1.0)
                painter.setPen(border_pen)
                painter.drawPath(path)
                
                # Text
                painter.setPen(self.theme.help_text)
                f = QFont()
                f.setPixelSize(14)
                painter.setFont(f)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, tip)

            painter.end()

        # ---------------------------- Interaction ----------------------------

        def _index_at_global_pos(self, pos) -> Optional[Tuple[int, int]]:
            for idx, layout in enumerate(self.regions):
                if layout.display_path.contains(pos):
                    char_index = layout.index_at(pos)
                    return idx, char_index
            return None

        def mouseMoveEvent(self, event):
            p = event.position()
            hit = self._index_at_global_pos(p)
            self.hovered_index = (hit[0] if hit else None)
            if self.is_selecting and hit:
                self.selection_end = hit
                if self.debug and hit != self.selection_start:
                    box_idx, char_idx = hit
                    debug_log(self.debug, f"Selection end: box={box_idx} char={char_idx}")
            self.update()

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                p = event.position()
                hit = self._index_at_global_pos(p)
                if hit:
                    self.is_selecting = True
                    self.selection_start = hit
                    self.selection_end = hit
                    if self.debug:
                        box_idx, char_idx = hit
                        debug_log(self.debug, f"Selection start: box={box_idx} char={char_idx} at pos({p.x():.1f}, {p.y():.1f})")
                    self.update()

        def mouseReleaseEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                self.is_selecting = False
                if self.selection_start and self.selection_end:
                    normalized = self._normalize_selection_bounds()
                    if normalized:
                        (start_box, start_char), (end_box, end_char) = normalized
                        parts: List[str] = []
                        for i in range(start_box, end_box + 1):
                            layout = self.regions[i]
                            t = layout.text
                            if not t:
                                continue
                            
                            if i == start_box and i == end_box:
                                # Single region: apply smart word snapping
                                snapped_start, snapped_end = layout.snap_to_word_boundaries(start_char, end_char)
                                parts.append(t[snapped_start:snapped_end])
                            elif i == start_box:
                                # First region: snap start to word boundary
                                word = layout.get_word_at(start_char)
                                actual_start = word[0] if word else start_char
                                parts.append(t[actual_start:])
                            elif i == end_box:
                                # Last region: snap end to word boundary
                                word = layout.get_word_at(max(0, end_char - 1)) if end_char > 0 else None
                                actual_end = word[1] if word else end_char
                                parts.append(t[:actual_end])
                            else:
                                # Middle regions: entire text
                                parts.append(t)
                        self.selected_text = " ".join(p for p in parts if p)
                        if self.selected_text:
                            try:
                                subprocess.run(["wl-copy"], input=self.selected_text.encode(), check=True)
                            except subprocess.CalledProcessError as exc:
                                notify("Clipboard Error", f"Failed to copy text: {exc}")
                            QApplication.quit()

        def _normalize_selection_bounds(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
            if not self.selection_start or not self.selection_end:
                return None
            (sb, sc), (eb, ec) = self.selection_start, self.selection_end
            if (eb < sb) or (eb == sb and ec < sc):
                sb, sc, eb, ec = eb, ec, sb, sc
            return (sb, sc), (eb, ec)

        def keyPressEvent(self, event):
            if event.key() == Qt.Key.Key_Escape:
                self.was_cancelled = True
                QApplication.quit()

        def closeEvent(self, event):
            try:
                self.scan_timer.stop()
                self.fade_timer.stop()
                self.cleanup_timer.stop()
                if not self.was_cancelled and self.selected_text is None and self.total_regions > 0:
                    self.was_cancelled = True
                if hasattr(self, "ocr_worker") and self.ocr_worker.isRunning():
                    self.ocr_worker.requestInterruption()
                    self.ocr_worker.wait(500)
            finally:
                super().closeEvent(event)

    # ------------------------------- Run UI -------------------------------

    app = QApplication.instance() or QApplication(sys.argv)
    debug_log(debug, "Launching overlay UI")
    overlay = OverlayWindow(image_path, fast, debug, live)
    overlay.show()
    app.exec()
    debug_log(debug, f"Overlay session complete (regions={overlay.total_regions})")
    overlay.close()

    # Cleanup temp shot
    try:
        Path(image_path).unlink(missing_ok=True)  # py3.8+: ignore if not present
    except Exception:
        pass

    if overlay.ocr_error:
        raise RuntimeError(overlay.ocr_error)

    return overlay.selected_text, overlay.total_regions, overlay.was_cancelled
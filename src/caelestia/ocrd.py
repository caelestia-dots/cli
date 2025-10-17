#!/usr/bin/env python3
"""
Caelestia OCR Daemon (ocrd)

A persistent daemon that keeps OCR models hot in memory for fast text detection.
Uses RapidOCR with ONNXRuntime for optimal CPU performance.

Future-ready for NPU (XDNA) acceleration when AMD's ONNX Runtime EP is stable on Linux.
"""

import json
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

# Limit thread contention
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")

try:
    from threadpoolctl import ThreadpoolController
except ImportError:  # pragma: no cover - optional performance tuning
    ThreadpoolController = None

# Import RapidOCR
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    print("Error: rapidocr-onnxruntime not installed.", file=sys.stderr)
    print("Install with: pip install rapidocr-onnxruntime", file=sys.stderr)
    sys.exit(1)


class PerformanceManager:
    """Manage dynamic CPU thread and affinity settings."""

    def __init__(self, config: Dict):
        self._config = config or {}
        self._perf_config = self._config.get("performance", {})
        self._thread_controller = ThreadpoolController() if ThreadpoolController else None
        self._affinity_supported = hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity")
        self._available_cpus = self._detect_cpu_pool()
        self._thread_counts = self._resolve_thread_counts()
        self._affinity_sets = self._resolve_affinity_sets()

    def describe(self) -> str:
        total_cpus = len(self._available_cpus)
        if self._thread_controller:
            threads_info = (
                f"threads idle/standard/fast="
                f"{self._thread_counts['idle']}/"
                f"{self._thread_counts['standard']}/"
                f"{self._thread_counts['fast']}"
            )
        else:
            threads_info = "threads control=disabled (threadpoolctl missing)"

        if self._affinity_supported and total_cpus:
            affinity_info = (
                f"cores idle/standard/fast="
                f"{len(self._affinity_sets['idle'])}/"
                f"{len(self._affinity_sets['standard'])}/"
                f"{len(self._affinity_sets['fast'])} of {total_cpus}"
            )
        else:
            affinity_info = "cores control=disabled"

        return f"{threads_info}; {affinity_info}"

    def apply_idle(self) -> None:
        """Clamp affinity and threads to idle settings."""
        self._set_affinity(self._affinity_sets.get("idle"))
        self._set_threads(self._thread_counts.get("idle"))

    def boost(self, fast_mode: bool) -> tuple[object | None, object | None]:
        """Boost resources for active OCR work."""
        mode = "fast" if fast_mode else "standard"
        thread_state = self._set_threads(self._thread_counts.get(mode), track_previous=True)
        affinity_state = self._set_affinity(self._affinity_sets.get(mode), track_previous=True)
        return thread_state, affinity_state

    def restore(self, thread_state: object | None, affinity_state: object | None) -> None:
        """Restore previous thread and affinity settings."""
        if thread_state is not None and self._thread_controller:
            with suppress(Exception):
                self._thread_controller.limit(limits=thread_state)

        if affinity_state is not None and self._affinity_supported:
            with suppress(Exception):
                os.sched_setaffinity(0, affinity_state)

    def _detect_cpu_pool(self) -> List[int]:
        if self._affinity_supported:
            with suppress(Exception):
                return sorted(os.sched_getaffinity(0))  # type: ignore[arg-type]
        count = os.cpu_count() or 1
        return list(range(count))

    def _resolve_thread_counts(self) -> Dict[str, int]:
        total_threads = max(len(self._available_cpus), os.cpu_count() or 1)

        def _clamp(value, fallback):
            if value is None:
                return fallback
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return fallback
            if parsed <= 0:
                return fallback
            return max(1, min(total_threads, parsed))

        idle_default = max(1, total_threads // 4) or 1
        standard_default = max(1, min(total_threads, max(idle_default, total_threads // 2)))
        fast_default = max(1, total_threads)

        return {
            "idle": _clamp(self._perf_config.get("idle_threads"), idle_default),
            "standard": _clamp(self._perf_config.get("standard_threads"), standard_default),
            "fast": _clamp(self._perf_config.get("fast_threads"), fast_default),
        }

    def _resolve_affinity_sets(self) -> Dict[str, set[int]]:
        total = len(self._available_cpus)
        if total == 0 or not self._affinity_supported:
            return {"idle": set(), "standard": set(), "fast": set(), "all": set()}

        def _clamp(value, fallback):
            if value is None:
                return fallback
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return fallback
            if parsed <= 0:
                return fallback
            return max(1, min(total, parsed))

        idle_count = _clamp(self._perf_config.get("idle_cores"), max(1, total // 4) or 1)
        standard_count = _clamp(
            self._perf_config.get("standard_cores"),
            max(idle_count, min(total, max(1, total // 2))),
        )
        fast_count = _clamp(self._perf_config.get("fast_cores"), total)

        cores = self._available_cpus

        def _slice(count: int) -> set[int]:
            if count >= total:
                return set(cores)
            return set(cores[:count])

        return {
            "all": set(cores),
            "idle": _slice(idle_count),
            "standard": _slice(standard_count),
            "fast": _slice(fast_count),
        }

    def _set_threads(self, target: int | None, track_previous: bool = False) -> object | None:
        if not self._thread_controller or not target:
            return None
        try:
            previous = self._thread_controller.limit(limits=target)
        except Exception:
            return None
        return previous if track_previous else None

    def _set_affinity(self, target: set[int] | None, track_previous: bool = False) -> object | None:
        if not self._affinity_supported or not target:
            return None
        try:
            current = os.sched_getaffinity(0)  # type: ignore[attr-defined]
            if current == target:
                return current if track_previous else None
            os.sched_setaffinity(0, target)  # type: ignore[arg-type]
            return current if track_previous else None
        except Exception:
            return None


class OCRDaemon:
    """Persistent OCR service with hot model cache."""
    
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.config = self._load_config()
        self.ocr_engine = None
        self.stats = {
            "requests": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "warmed": False,
            "last_warm": 0.0
        }
        self.performance = PerformanceManager(self.config)
        self.performance.apply_idle()
        cpu_count = os.cpu_count() or 4
        default_workers = max(1, cpu_count // 2)
        self.stream_workers = max(1, min(8, default_workers))
    
    def _load_config(self) -> Dict:
        """Load OCR configuration."""
        config_dir = Path.home() / ".config" / "caelestia"
        config_file = config_dir / "ocr.json"
        
        default_config = {
            "provider": "cpu-ort",  # cpu-ort, gpu-rocm, npu-xdna
            "downscale": 0.6,       # Detection downscale factor
            "tiles": 1,             # Number of tiles for parallel processing
            "max_boxes": 300,       # Maximum boxes to return
            "use_gpu": False,       # Use GPU if available (experimental)
            "warm_start": True,     # Run warm-up inference on start
            "performance": {}       # Thread/affinity tuning
        }
        
        try:
            if config_file.exists():
                with open(config_file) as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
        except Exception as e:
            print(f"Warning: Could not load config: {e}", file=sys.stderr)
        
        return default_config
    
    def _init_ocr(self):
        """Initialize RapidOCR engine with warm-up."""
        print("Initializing RapidOCR engine...")
        start = time.time()
        
        # Initialize with GPU if configured (experimental on AMD)
        use_gpu = self.config.get("use_gpu", False)
        self.ocr_engine = RapidOCR(use_cuda=use_gpu)
        
        # Warm-up: run inference on a tiny image to initialize ONNX graph
        if self.config.get("warm_start", True):
            self._run_warm_up("startup")

        elapsed = time.time() - start
        print(f"OCR engine ready in {elapsed:.2f}s")
        print(f"Performance profile: {self.performance.describe()}")
    
    def _downscale_for_detection(self, img: Image.Image, factor: float) -> Tuple[Image.Image, float]:
        """
        Downscale image for faster detection.
        Returns (downscaled_image, scale_factor_applied)
        """
        if factor >= 1.0:
            return img, 1.0
        
        w, h = img.size
        new_w = int(w * factor)
        new_h = int(h * factor)
        
        # Use high-quality downsampling
        downscaled = img.resize((new_w, new_h), Image.LANCZOS)
        return downscaled, factor
    
    def _rescale_boxes(self, boxes: List, scale_factor: float) -> List:
        """Rescale bounding boxes back to original image coordinates."""
        if scale_factor >= 1.0:
            return boxes

        rescaled = []
        for box in boxes:
            # box is [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
            rescaled_box = [[int(x / scale_factor), int(y / scale_factor)] for x, y in box]
            rescaled.append(rescaled_box)

        return rescaled

    def _prepare_image(self, image_path: str, fast_mode: bool) -> tuple[Image.Image, Image.Image, np.ndarray, float, Dict[str, float]]:
        """Load an image from disk and prepare downscaled numpy array for OCR."""
        load_start = time.time()
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        load_time = time.time() - load_start

        downscale_start = time.time()
        downscale_factor = self.config.get("downscale", 0.6)
        if fast_mode:
            downscale_factor = min(downscale_factor, 0.5)
        img_for_ocr, actual_scale = self._downscale_for_detection(img, downscale_factor)
        downscale_time = time.time() - downscale_start

        img_array = np.array(img_for_ocr)

        return img, img_for_ocr, img_array, actual_scale, {
            "load": load_time,
            "downscale": downscale_time,
        }
    
    def process_image(self, image_path: str, fast_mode: bool = False) -> Dict:
        """
        Process an image and return OCR results.
        
        Args:
            image_path: Path to the image file
            fast_mode: Enable aggressive optimizations
            
        Returns:
            Dict with keys: boxes, texts, scores, timing
        """
        start_time = time.time()
        
        try:
            original_img, img_for_ocr, img_array, actual_scale, timing = self._prepare_image(image_path, fast_mode)
            load_time = timing["load"]
            downscale_time = timing["downscale"]
            
            # Run OCR
            boost_state = self.performance.boost(fast_mode)
            ocr_time = 0.0
            result = None
            elapsed = 0.0
            try:
                ocr_start = time.time()
                result, elapsed = self.ocr_engine(img_array)
                ocr_time = time.time() - ocr_start
            finally:
                self.performance.restore(*boost_state)
                self.performance.apply_idle()
            
            # Parse results
            if result is None or len(result) == 0:
                boxes, texts, scores = [], [], []
            else:
                # RapidOCR returns: (result_list, elapsed) where result_list = [[bbox, text, score], ...]
                boxes = [item[0] for item in result]
                texts = [item[1] for item in result]
                scores = [item[2] for item in result]
                
                # Rescale boxes back to original coordinates
                boxes = self._rescale_boxes(boxes, actual_scale)
            
            # Limit boxes if configured
            max_boxes = self.config.get("max_boxes", 300)
            if fast_mode:
                max_boxes = min(max_boxes, 150)
            
            if len(boxes) > max_boxes:
                # Sort by score and keep top N
                sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                sorted_indices = sorted_indices[:max_boxes]
                boxes = [boxes[i] for i in sorted_indices]
                texts = [texts[i] for i in sorted_indices]
                scores = [scores[i] for i in sorted_indices]
            
            total_time = time.time() - start_time
            
            # Update stats
            self.stats["requests"] += 1
            self.stats["total_time"] += total_time
            self.stats["avg_time"] = self.stats["total_time"] / self.stats["requests"]
            self.stats["warmed"] = True
            self.stats["last_warm"] = time.time()

            return {
                "status": "success",
                "boxes": boxes,
                "texts": texts,
                "scores": scores,
                "timing": {
                    "load": round(load_time * 1000, 2),
                    "downscale": round(downscale_time * 1000, 2),
                    "ocr": round(ocr_time * 1000, 2),
                    "total": round(total_time * 1000, 2)
                },
                "image_size": f"{original_img.size[0]}x{original_img.size[1]}",
                "processed_size": f"{img_for_ocr.size[0]}x{img_for_ocr.size[1]}",
                "num_detections": len(boxes)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timing": {
                    "total": round((time.time() - start_time) * 1000, 2)
                }
            }

    def _detect_regions_for_stream(
        self,
        img_array: np.ndarray,
    ) -> tuple[list[list[list[float]]], list[np.ndarray], float]:
        """Run detection pipeline to obtain candidate regions and cropped images."""
        if self.ocr_engine is None:
            raise RuntimeError("OCR engine not initialised")

        raw_h, raw_w = img_array.shape[:2]
        op_record: Dict[str, Any] = {}
        proc_img, ratio_h, ratio_w = self.ocr_engine.preprocess(img_array)
        op_record["preprocess"] = {"ratio_h": ratio_h, "ratio_w": ratio_w}

        proc_img, op_record = self.ocr_engine.maybe_add_letterbox(proc_img, op_record)
        dt_boxes, det_elapsed = self.ocr_engine.auto_text_det(proc_img)

        if dt_boxes is None or len(dt_boxes) == 0:
            return [], [], det_elapsed

        if isinstance(dt_boxes, np.ndarray):
            boxes_array = dt_boxes
        else:
            boxes_array = np.array(dt_boxes)

        sorted_boxes = self.ocr_engine.sorted_boxes(boxes_array)
        crop_list = self.ocr_engine.get_crop_img_list(proc_img, sorted_boxes)

        origin_boxes = self.ocr_engine._get_origin_points(sorted_boxes, op_record, raw_h, raw_w)
        origin_boxes_list = origin_boxes.astype(float).tolist()

        return origin_boxes_list, crop_list, det_elapsed

    def _recognize_region(self, crop_img: np.ndarray) -> tuple[str, float]:
        """Recognize text inside a cropped region."""
        if self.ocr_engine is None:
            raise RuntimeError("OCR engine not initialised")

        images: list[np.ndarray] = [crop_img]
        if self.ocr_engine.use_cls:
            images, _cls_res, _cls_time = self.ocr_engine.text_cls(images)

        rec_res, _rec_time = self.ocr_engine.text_rec(images, False)
        if not rec_res:
            return "", 0.0

        text_entry = rec_res[0]
        if isinstance(text_entry, (list, tuple)) and len(text_entry) >= 2:
            text = text_entry[0]
            score = float(text_entry[1])
        else:
            text = str(text_entry)
            score = 0.0

        return text, score

    @staticmethod
    def _compute_bbox(box: list[list[float]]) -> tuple[float, float, float, float]:
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def stream_image(self, image_path: str, fast_mode: bool, send) -> None:
        """Stream OCR results incrementally via the provided send callback."""
        start_time = time.time()

        if not image_path:
            raise ValueError("Missing image path for streaming request")

        original_img, img_for_ocr, img_array, actual_scale, timing = self._prepare_image(image_path, fast_mode)

        boost_state = self.performance.boost(fast_mode)
        detected_boxes: list[list[list[float]]] = []
        crop_list: list[np.ndarray] = []
        det_elapsed = 0.0
        recognition_start = time.time()
        emitted = 0

        def _scale_box(box: list[list[float]]) -> list[list[float]]:
            if actual_scale >= 1.0:
                return [[float(x), float(y)] for x, y in box]
            return [[float(x / actual_scale), float(y / actual_scale)] for x, y in box]

        try:
            detected_boxes, crop_list, det_elapsed = self._detect_regions_for_stream(img_array)
            scaled_boxes = [_scale_box(box) for box in detected_boxes]

            send(
                {
                    "type": "det",
                    "boxes": scaled_boxes,
                    "image_size": [original_img.size[0], original_img.size[1]],
                    "processed_size": [img_for_ocr.size[0], img_for_ocr.size[1]],
                    "timing": {
                        "load": round(timing["load"] * 1000, 2),
                        "downscale": round(timing["downscale"] * 1000, 2),
                        "det": round(det_elapsed * 1000, 2),
                    },
                }
            )

            if not detected_boxes:
                send(
                    {
                        "type": "done",
                        "emitted": 0,
                        "detected": 0,
                        "timing": {
                            "total": round((time.time() - start_time) * 1000, 2),
                        },
                    }
                )
                return

            threshold = getattr(self.ocr_engine, "text_score", 0.0)

            def worker(idx: int, crop: np.ndarray, box_scaled: list[list[float]]):
                try:
                    text, score = self._recognize_region(crop)
                except Exception as exc:  # noqa: BLE001
                    return idx, box_scaled, "", 0.0, str(exc)
                return idx, box_scaled, text, float(score), None

            with ThreadPoolExecutor(max_workers=self.stream_workers) as executor:
                futures = [
                    executor.submit(worker, idx, crop, scaled_boxes[idx])
                    for idx, crop in enumerate(crop_list)
                ]

                for future in as_completed(futures):
                    idx, box_scaled, text, score, error = future.result()
                    if error is not None:
                        send({"type": "error", "message": error, "index": idx})
                        continue

                    if not text.strip() or score < threshold:
                        continue

                    bbox = self._compute_bbox(box_scaled)
                    send(
                        {
                            "type": "update",
                            "index": idx,
                            "box": box_scaled,
                            "bbox": [float(v) for v in bbox],
                            "text": text,
                            "conf": float(score),
                        }
                    )
                    emitted += 1
        except Exception as exc:  # noqa: BLE001
            send({"type": "error", "message": str(exc)})
            send(
                {
                    "type": "done",
                    "emitted": emitted,
                    "detected": len(detected_boxes),
                    "timing": {
                        "total": round((time.time() - start_time) * 1000, 2),
                        "rec": round((time.time() - recognition_start) * 1000, 2),
                    },
                    "error": True,
                }
            )
            return
        else:
            send(
                {
                    "type": "done",
                    "emitted": emitted,
                    "detected": len(detected_boxes),
                    "timing": {
                        "total": round((time.time() - start_time) * 1000, 2),
                        "rec": round((time.time() - recognition_start) * 1000, 2),
                    },
                }
            )
        finally:
            self.performance.restore(*boost_state)
            self.performance.apply_idle()

    def _run_warm_up(self, reason: str, fast_mode: bool = False) -> Dict:
        """Execute a warm-up inference to keep models hot."""
        if self.ocr_engine is None:
            raise RuntimeError("OCR engine not initialised")

        print(f"Warm-up inference ({reason}, fast={fast_mode})")
        start = time.time()
        boost_state = self.performance.boost(fast_mode)
        try:
            side = 160 if fast_mode else 224
            dummy_img = np.ones((side, side, 3), dtype=np.uint8) * 255
            _result, _elapsed = self.ocr_engine(dummy_img)
            duration = time.time() - start
            self.stats["warmed"] = True
            self.stats["last_warm"] = time.time()
            return {
                "status": "success",
                "timing": {
                    "warm": round(duration * 1000, 2)
                }
            }
        except Exception as exc:
            print(f"Warning: Warm-up failed: {exc}", file=sys.stderr)
            return {
                "status": "error",
                "error": str(exc)
            }
        finally:
            self.performance.restore(*boost_state)
            self.performance.apply_idle()

    def start(self):
        """Start the daemon and listen for requests."""
        # Create socket directory
        socket_dir = Path(self.socket_path).parent
        socket_dir.mkdir(parents=True, exist_ok=True)
        
        # Remove stale socket
        if Path(self.socket_path).exists():
            Path(self.socket_path).unlink()
        
        # Initialize OCR engine
        self._init_ocr()
        
        # Create UNIX domain socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.socket_path)
        sock.listen(5)
        
        print(f"OCR Daemon listening on {self.socket_path}")
        print(f"Config: {self.config}")
        print("Ready to process requests...")
        
        try:
            while True:
                conn, _ = sock.accept()
                try:
                    # Receive request
                    data = b""
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                        if b"\n" in chunk:
                            break
                    
                    if not data:
                        continue
                    
                    request = json.loads(data.decode())
                    cmd = request.get("cmd")
                    
                    if cmd == "ocr_full":
                        image_path = request.get("path")
                        fast_mode = request.get("fast", False)
                        
                        print(f"Processing: {image_path} (fast={fast_mode})")
                        result = self.process_image(image_path, fast_mode)
                        
                        if result["status"] == "success":
                            print(f"  → {result['num_detections']} detections in {result['timing']['total']}ms")
                        else:
                            print(f"  → Error: {result.get('error')}")
                        
                        # Send response
                        response = json.dumps(result) + "\n"
                        conn.sendall(response.encode())

                    elif cmd == "stream_ocr":
                        image_path = request.get("path")
                        fast_mode = request.get("fast", False)

                        print(f"Streaming: {image_path} (fast={fast_mode})")
                        writer = conn.makefile("w")

                        def send_stream(payload: Dict[str, Any]) -> None:
                            writer.write(json.dumps(payload) + "\n")
                            writer.flush()

                        try:
                            self.stream_image(image_path, fast_mode, send_stream)
                        except Exception as exc:  # noqa: BLE001
                            print(f"  → Stream error: {exc}", file=sys.stderr)
                            try:
                                send_stream({"type": "error", "message": str(exc)})
                                send_stream(
                                    {
                                        "type": "done",
                                        "emitted": 0,
                                        "detected": 0,
                                        "timing": {
                                            "total": 0.0,
                                        },
                                        "error": True,
                                    }
                                )
                            except Exception:
                                pass
                        finally:
                            try:
                                writer.close()
                            except Exception:
                                pass

                    elif cmd == "stats":
                        response = json.dumps(self.stats) + "\n"
                        conn.sendall(response.encode())
                    
                    elif cmd == "ping":
                        response = json.dumps({"status": "ok"}) + "\n"
                        conn.sendall(response.encode())

                    elif cmd == "warm_up":
                        fast_mode = request.get("fast", False)
                        result = self._run_warm_up("remote", fast_mode)
                        response = json.dumps(result) + "\n"
                        conn.sendall(response.encode())

                    else:
                        response = json.dumps({"status": "error", "error": f"Unknown command: {cmd}"}) + "\n"
                        conn.sendall(response.encode())
                
                except Exception as e:
                    print(f"Error handling request: {e}", file=sys.stderr)
                    try:
                        response = json.dumps({"status": "error", "error": str(e)}) + "\n"
                        conn.sendall(response.encode())
                    except:
                        pass
                finally:
                    conn.close()
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            sock.close()
            if Path(self.socket_path).exists():
                Path(self.socket_path).unlink()


def main():
    """Main entry point for the daemon."""
    socket_path = os.environ.get("CAELESTIA_OCR_SOCKET", "/tmp/caelestia_ocrd.sock")

    daemon = OCRDaemon(socket_path)
    daemon.start()


if __name__ == "__main__":
    main()

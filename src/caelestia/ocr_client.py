"""
OCR Client for communicating with the OCR daemon.
"""

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple


class StreamNotSupportedError(RuntimeError):
    """Raised when the connected daemon does not support streaming."""


class OCRClient:
    """Client for communicating with the OCR daemon."""
    
    def __init__(self):
        socket_env = os.environ.get("CAELESTIA_OCR_SOCKET")
        primary_socket = Path(socket_env) if socket_env else Path("/tmp/caelestia_ocrd.sock")
        legacy_socket = Path.home() / ".cache" / "caelestia" / "ocrd.sock"

        self._socket_candidates = [primary_socket]
        if legacy_socket != primary_socket:
            self._socket_candidates.append(legacy_socket)

        self.socket_path = str(primary_socket)
        self.daemon_started = False

    def _existing_socket(self) -> Optional[str]:
        """Return the first available socket path, if any."""
        for candidate in self._socket_candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _refresh_socket_path(self) -> None:
        """Update active socket path if a candidate exists."""
        existing = self._existing_socket()
        if existing:
            self.socket_path = existing
    
    def _ensure_daemon(self) -> bool:
        """Ensure the OCR daemon is running."""
        # Check if socket exists and is responsive
        existing_socket = self._existing_socket()
        if existing_socket:
            try:
                response = self._send_request({"cmd": "ping"}, timeout=1.0, socket_override=existing_socket)
                if response.get("status") == "ok":
                    self.socket_path = existing_socket
                    return True
            except Exception:
                pass

        # Try to start daemon via systemd
        try:
            result = subprocess.run(
                ["systemctl", "--user", "start", "caelestia-ocrd"],
                capture_output=True,
                timeout=2
            )
            if result.returncode == 0:
                # Wait for socket to appear
                for _ in range(10):
                    self._refresh_socket_path()
                    current_socket = self._existing_socket()
                    if current_socket:
                        time.sleep(0.2)  # Give daemon time to initialize
                        try:
                            response = self._send_request({"cmd": "ping"}, timeout=1.0, socket_override=current_socket)
                            if response.get("status") == "ok":
                                self.socket_path = current_socket
                                self.daemon_started = True
                                return True
                        except Exception:
                            pass
                    time.sleep(0.1)
        except Exception:
            pass

        # Fall back to starting daemon directly
        try:
            import sys
            
            # Start daemon in background
            daemon_script = Path(__file__).parent / "ocrd.py"
            subprocess.Popen(
                [sys.executable, str(daemon_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            
            # Wait for daemon to be ready
            for _ in range(20):
                self._refresh_socket_path()
                current_socket = self._existing_socket()
                if current_socket:
                    time.sleep(0.2)
                    try:
                        response = self._send_request({"cmd": "ping"}, timeout=1.0, socket_override=current_socket)
                        if response.get("status") == "ok":
                            self.socket_path = current_socket
                            self.daemon_started = True
                            return True
                    except Exception:
                        pass
                time.sleep(0.2)
        except Exception as e:
            print(f"Failed to start OCR daemon: {e}")
            return False
        
        return False
    
    def _send_request(self, request: Dict, timeout: float = 30.0, socket_override: Optional[str] = None) -> Dict:
        """Send a request to the daemon and get response."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            target_path = socket_override or self.socket_path
            sock.connect(target_path)
            
            # Send request
            request_data = json.dumps(request) + "\n"
            sock.sendall(request_data.encode())
            
            # Receive response
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in chunk:
                    break
            
            response = json.loads(data.decode())
            return response
        finally:
            sock.close()
    
    def ocr_full(self, image_path: str, fast: bool = False) -> Tuple[List, List, List]:
        """
        Run OCR on an image.
        
        Args:
            image_path: Path to the image file
            fast: Enable fast mode with aggressive optimizations
            
        Returns:
            Tuple of (boxes, texts, scores) where:
            - boxes: List of bounding box coordinates [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
            - texts: List of detected text strings
            - scores: List of confidence scores (0-1)
        """
        # Ensure daemon is running
        if not self._ensure_daemon():
            raise RuntimeError(
                "Could not start OCR daemon. "
                "Please install dependencies: pip install rapidocr-onnxruntime"
            )
        
        # Send OCR request
        request = {
            "cmd": "ocr_full",
            "path": str(image_path),
            "fast": fast
        }
        
        response = self._send_request(request, timeout=30.0)
        
        if response.get("status") != "success":
            error = response.get("error", "Unknown error")
            raise RuntimeError(f"OCR failed: {error}")
        
        boxes = response.get("boxes", [])
        texts = response.get("texts", [])
        scores = response.get("scores", [])

        return boxes, texts, scores

    def stream_ocr(self, image_path: str, fast: bool = False) -> Generator[Dict, None, None]:
        """Yield streaming OCR messages from the daemon."""
        if not self._ensure_daemon():
            raise RuntimeError(
                "Could not start OCR daemon. "
                "Please install dependencies: pip install rapidocr-onnxruntime"
            )

        request = {
            "cmd": "stream_ocr",
            "path": str(image_path),
            "fast": fast,
        }

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.socket_path)
            sock.sendall((json.dumps(request) + "\n").encode())

            reader = sock.makefile("r")
            try:
                while True:
                    line = reader.readline()
                    if not line:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "status" in message and "type" not in message:
                        error = message.get("error", "Streaming not supported")
                        if "Unknown command" in error:
                            raise StreamNotSupportedError(error)
                        raise RuntimeError(error)

                    yield message

                    if message.get("type") == "done":
                        break
            finally:
                reader.close()
        finally:
            sock.close()

    def warm_up(self, fast: bool = False) -> Dict:
        """Ask daemon to run a warm-up inference."""
        if not self._ensure_daemon():
            raise RuntimeError(
                "Could not start OCR daemon. "
                "Please install dependencies: pip install rapidocr-onnxruntime"
            )

        request = {
            "cmd": "warm_up",
            "fast": fast
        }

        response = self._send_request(request, timeout=10.0)

        if response.get("status") != "success":
            error = response.get("error", "Unknown error")
            raise RuntimeError(f"Warm-up failed: {error}")

        return response

    def get_stats(self) -> Optional[Dict]:
        """Get daemon statistics."""
        try:
            if self._ensure_daemon():
                return self._send_request({"cmd": "stats"}, timeout=1.0)
        except Exception:
            pass
        return None


# Global client instance
_client = None

def get_ocr_client() -> OCRClient:
    """Get or create the global OCR client instance."""
    global _client
    if _client is None:
        _client = OCRClient()
    return _client


def ocr_full(image_path: str, fast: bool = False) -> Tuple[List, List, List]:
    """
    Run OCR on an image (convenience function).

    Args:
        image_path: Path to the image file
        fast: Enable fast mode

    Returns:
        Tuple of (boxes, texts, scores)
    """
    client = get_ocr_client()
    return client.ocr_full(image_path, fast)


def stream_ocr(image_path: str, fast: bool = False) -> Generator[Dict, None, None]:
    """Convenience wrapper for streaming OCR messages."""
    client = get_ocr_client()
    return client.stream_ocr(image_path, fast)

import json
import re
import shutil
import subprocess
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from caelestia.utils.notify import close_notification, notify
from caelestia.utils.paths import (
    recording_notif_path,
    recording_path,
    recordings_dir,
    user_config_path,
)

RECORDER = "gpu-screen-recorder"

AUDIO_MODES = {
    "mic": "default_input",
    "system": "default_output",
    "combined": "default_output|default_input",
}

# Maximum time (in seconds) to wait for the recorder process to exit cleanly
STOP_TIMEOUT = 5.0
STOP_POLL_INTERVAL = 0.1


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if getattr(self.args, "status", False):
            self.status()
        elif getattr(self.args, "stop", False):
            self.stop()
        elif self.args.pause:
            subprocess.run(
                ["pkill", "-USR2", "-f", RECORDER], stdout=subprocess.DEVNULL
            )
        elif self.proc_running():
            self.stop()
        else:
            self.start()

    def status(self) -> None:
        """Print the current recording status."""
        if self.proc_running():
            print("Recording: RUNNING")
        else:
            print("Recording: STOPPED")

    def proc_running(self) -> bool:
        return (
            subprocess.run(["pidof", RECORDER], stdout=subprocess.DEVNULL).returncode
            == 0
        )

    def intersects(
        self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]
    ) -> bool:
        return (
            a[0] < b[0] + b[2]
            and a[0] + a[2] > b[0]
            and a[1] < b[1] + b[3]
            and a[1] + a[3] > b[1]
        )

    def get_audio_device(self, audio_mode: str | None) -> str:
        """Return the audio device string for the given mode, with fallback handling.

        Returns an empty string when no audio should be recorded.
        """
        if not audio_mode or audio_mode == "none":
            return ""

        device = AUDIO_MODES.get(audio_mode, "")

        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True,
                text=True,
                check=True,
            )
            available_devices = [
                line.split("\t")[1]
                for line in result.stdout.strip().split("\n")
                if line
            ]

            if device and device not in available_devices:
                print(
                    f"Warning: audio device '{device}' not available, falling back to default"
                )
                if audio_mode == "mic":
                    candidates = [
                        d
                        for d in available_devices
                        if "input" in d.lower() or "mic" in d.lower()
                    ]
                    device = candidates[0] if candidates else ""
                elif audio_mode == "system":
                    candidates = [
                        d
                        for d in available_devices
                        if "output" in d.lower() or "monitor" in d.lower()
                    ]
                    device = candidates[0] if candidates else ""

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: could not check audio devices, audio recording may fail")

        return device

    def get_window_region(self) -> str | None:
        """Select a window via slurp and return its region string."""
        try:
            clients = json.loads(subprocess.check_output(["hyprctl", "clients", "-j"]))

            if not clients:
                print("No windows found")
                return None

            slurp_regions = [
                f"{c['at'][0]},{c['at'][1]} {c['size'][0]}x{c['size'][1]}"
                for c in clients
            ]

            result = subprocess.run(
                ["slurp", "-f", "%wx%h+%x+%y"],
                input="\n".join(slurp_regions),
                capture_output=True,
                text=True,
            )

            return result.stdout.strip() if result.returncode == 0 else None

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            print(f"Error getting window region: {e}")
            return None

    def _max_refresh_rate_for_region(
        self,
        monitors: list[dict],
        region: tuple[int, int, int, int],
    ) -> int:
        """Return the highest refresh rate among monitors that overlap *region*."""
        max_rr = 0
        for monitor in monitors:
            if self.intersects(
                (monitor["x"], monitor["y"], monitor["width"], monitor["height"]),
                region,
            ):
                max_rr = max(max_rr, round(monitor["refreshRate"]))
        return max_rr

    def _parse_region(self, region_str: str) -> tuple[int, int, int, int]:
        """Parse a ``WxH+X+Y`` region string into ``(x, y, w, h)``."""
        m = re.match(r"(\d+)x(\d+)\+(\d+)\+(\d+)", region_str)
        if not m:
            raise ValueError(f"Invalid region format: {region_str!r}")
        w, h, x, y = map(int, m.groups())
        return x, y, w, h

    def start(self) -> None:
        args = ["-w"]

        video_mode = getattr(self.args, "mode", "fullscreen")
        audio_mode = getattr(self.args, "audio", None)

        monitors = json.loads(subprocess.check_output(["hyprctl", "monitors", "-j"]))

        # --- Video mode ---
        if video_mode == "region" or self.args.region:
            if self.args.region == "slurp" or not self.args.region:
                region_str = subprocess.check_output(
                    ["slurp", "-f", "%wx%h+%x+%y"], text=True
                ).strip()
            else:
                region_str = self.args.region.strip()

            x, y, w, h = self._parse_region(region_str)
            max_rr = self._max_refresh_rate_for_region(monitors, (x, y, w, h))
            args += ["region", "-region", region_str, "-f", str(max_rr)]

        elif video_mode == "window":
            window_info = self.get_window_region()
            if not window_info:
                print("Window selection cancelled")
                return

            x, y, w, h = self._parse_region(window_info)
            max_rr = self._max_refresh_rate_for_region(monitors, (x, y, w, h))
            args += ["region", "-region", window_info, "-f", str(max_rr)]

        else:  # fullscreen
            focused = next((m for m in monitors if m["focused"]), None)
            if focused:
                args += [focused["name"], "-f", str(round(focused["refreshRate"]))]

        # --- Audio mode ---
        audio_device = self.get_audio_device(audio_mode)
        if audio_device:
            args += ["-a", audio_device, "-ac", "opus", "-ab", "192k"]
            print(f"Recording with audio: {audio_device} ({audio_mode})")
        elif getattr(self.args, "sound", False):
            args += ["-a", "default_output"]
        else:
            print("Recording without audio")

        # --- Extra args from user config ---
        try:
            config = json.loads(user_config_path.read_text())
            extra = config.get("record", {}).get("extraArgs", [])
            if not isinstance(extra, list):
                raise ValueError("Config option 'record.extraArgs' must be an array")
            args += extra
        except (json.JSONDecodeError, FileNotFoundError):
            pass

        # --- Launch recorder ---
        recording_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [RECORDER, *args, "-o", str(recording_path)], start_new_session=True
        )

        audio_label = audio_mode if audio_device else "no audio"
        mode_text = f"{video_mode} with {audio_label}"
        notif = notify("-p", "Recording started", f"Recording {mode_text}...")
        recording_notif_path.write_text(notif)

        try:
            if proc.wait(1) != 0:
                close_notification(notif)
                notify(
                    "Recording failed",
                    "An error occurred attempting to start recorder. "
                    f"Command `{' '.join(proc.args)}` failed with exit code {proc.returncode}",
                )
        except subprocess.TimeoutExpired:
            pass  # Still running — good

    def stop(self) -> None:
        subprocess.run(["pkill", "-f", RECORDER], stdout=subprocess.DEVNULL)

        # Wait up to STOP_TIMEOUT seconds for a clean exit
        max_polls = int(STOP_TIMEOUT / STOP_POLL_INTERVAL)
        for _ in range(max_polls):
            if not self.proc_running():
                break
            time.sleep(STOP_POLL_INTERVAL)

        if not recording_path.exists():
            print("Warning: no recording file found")
            try:
                close_notification(recording_notif_path.read_text())
            except IOError:
                pass
            return

        # Move to recordings folder with a timestamped name
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        new_path = recordings_dir / f"recording_{timestamp}.mp4"
        recordings_dir.mkdir(exist_ok=True, parents=True)
        shutil.move(recording_path, new_path)

        # Dismiss the "recording started" notification
        try:
            close_notification(recording_notif_path.read_text())
        except IOError:
            pass

        # Show completion notification in background (non-blocking)
        try:
            subprocess.Popen(
                [
                    "notify-send",
                    "-a",
                    "caelestia-cli",
                    "--action=watch=Watch",
                    "--action=open=Open",
                    "--action=delete=Delete",
                    "Recording stopped",
                    f"Recording saved in {new_path}",
                ],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Could not show notification: {e}")

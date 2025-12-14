import json
import re
import shutil
import subprocess
import time
from argparse import Namespace
from datetime import datetime

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
    "combined": "CombinedSink.monitor",
}


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if hasattr(self.args, "status") and self.args.status:
            self.status()
        elif hasattr(self.args, "stop") and self.args.stop:
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
        """Check and display recording status"""
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

    def get_audio_device(self, audio_mode: str) -> str:
        """Get the appropriate audio device for the given mode with fallback handling."""
        if not audio_mode:
            return "none"

        device = AUDIO_MODES.get(audio_mode, "")

        # Check if the device is available
        if audio_mode in ["mic", "system", "combined"]:
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
                        f"Warning: Audio device '{device}' not available, falling back to default"
                    )
                    if audio_mode == "mic":
                        input_devices = [
                            d
                            for d in available_devices
                            if "input" in d.lower() or "mic" in d.lower()
                        ]
                        device = input_devices[0] if input_devices else ""
                    elif audio_mode == "system":
                        output_devices = [
                            d
                            for d in available_devices
                            if "output" in d.lower() or "monitor" in d.lower()
                        ]
                        device = output_devices[0] if output_devices else ""
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(
                    "Warning: Could not check audio devices, audio recording may fail"
                )
                device = ""

        return device

    def get_window_region(self) -> str | None:
        """Get window region using Hyprland's client list and slurp for selection."""
        try:
            # Get all windows from Hyprland
            clients = json.loads(subprocess.check_output(["hyprctl", "clients", "-j"]))

            if not clients:
                print("No windows found")
                return None

            # Create slurp format strings for each window
            slurp_regions = []
            for client in clients:
                x = client["at"][0]
                y = client["at"][1]
                w = client["size"][0]
                h = client["size"][1]
                slurp_regions.append(f"{x},{y} {w}x{h}")

            # Use slurp with predefined regions to pick a window
            slurp_input = "\n".join(slurp_regions)
            result = subprocess.run(
                ["slurp", "-f", "%wx%h+%x+%y"],
                input=slurp_input,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return None

            return result.stdout.strip()

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            print(f"Error getting window region: {e}")
            return None

    def start(self) -> None:
        args = ["-w"]

        # Get video mode and audio mode from args
        video_mode = getattr(self.args, "mode", "fullscreen")
        audio_mode = getattr(self.args, "audio", "none")

        monitors = json.loads(subprocess.check_output(["hyprctl", "monitors", "-j"]))

        # Handle video modes
        if video_mode == "region" or self.args.region:
            if self.args.region == "slurp" or not self.args.region:
                region = subprocess.check_output(
                    ["slurp", "-f", "%wx%h+%x+%y"], text=True
                ).strip()
            else:
                region = self.args.region.strip()
            args += ["region", "-region", region]

            m = re.match(r"(\d+)x(\d+)\+(\d+)\+(\d+)", region)
            if not m:
                raise ValueError(f"Invalid region: {region}")

            w, h, x, y = map(int, m.groups())
            r = x, y, w, h
            max_rr = 0
            for monitor in monitors:
                if self.intersects(
                    (monitor["x"], monitor["y"], monitor["width"], monitor["height"]), r
                ):
                    rr = round(monitor["refreshRate"])
                    max_rr = max(max_rr, rr)
            args += ["-f", str(max_rr)]

        elif video_mode == "window":
            window_info = self.get_window_region()
            if not window_info:
                print("Window selection canceled")
                return

            args += ["region", "-region", window_info]
            m = re.match(r"(\d+)x(\d+)\+(\d+)\+(\d+)", window_info)
            if not m:
                raise ValueError(f"Invalid window region: {window_info}")

            w, h, x, y = map(int, m.groups())
            r = x, y, w, h

            # Calculate max refresh rate for the window region
            max_rr = 0
            for monitor in monitors:
                if self.intersects(
                    (
                        monitor["x"],
                        monitor["y"],
                        monitor["width"],
                        monitor["height"],
                    ),
                    r,
                ):
                    rr = round(monitor["refreshRate"])
                    max_rr = max(max_rr, rr)
            args += ["-f", str(max_rr)]

        else:  # fullscreen
            focused_monitor = next(
                (monitor for monitor in monitors if monitor["focused"]), None
            )
            if focused_monitor:
                args += [
                    focused_monitor["name"],
                    "-f",
                    str(round(focused_monitor["refreshRate"])),
                ]

        # Handle audio modes
        audio_device = self.get_audio_device(audio_mode)
        if audio_device:
            args += ["-a", audio_device, "-ac", "opus", "-ab", "192k"]
            print(f"Recording with audio: {audio_device} ({audio_mode})")
        elif hasattr(self.args, "sound") and self.args.sound:
            args += ["-a", "default_output"]
        else:
            print("Recording without audio")

        # Load extra args from config
        try:
            config = json.loads(user_config_path.read_text())
            if "record" in config and "extraArgs" in config["record"]:
                args += config["record"]["extraArgs"]
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        except TypeError as e:
            raise ValueError(
                f"Config option 'record.extraArgs' should be an array: {e}"
            )

        recording_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [RECORDER, *args, "-o", str(recording_path)], start_new_session=True
        )

        # Show notification with mode info
        mode_text = f"{video_mode} with {audio_mode if audio_device else 'no'} audio"
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
            pass

    def stop(self) -> None:
        # Start killing recording process
        subprocess.run(["pkill", "-f", RECORDER], stdout=subprocess.DEVNULL)

        # Wait for recording to finish to avoid corrupted video file
        max_wait = 50  # Max 5 seconds
        wait_count = 0
        while self.proc_running() and wait_count < max_wait:
            time.sleep(0.1)
            wait_count += 1

        # Check if file exists before trying to move it
        if not recording_path.exists():
            print("Warning: No recording file found")
            try:
                close_notification(recording_notif_path.read_text())
            except IOError:
                pass
            return

        # Move to recordings folder
        new_path = (
            recordings_dir
            / f"recording_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.mp4"
        )
        recordings_dir.mkdir(exist_ok=True, parents=True)
        shutil.move(recording_path, new_path)

        # Close start notification
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

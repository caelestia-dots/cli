import json
import re
import shutil
import subprocess
import time
from argparse import Namespace
from datetime import datetime

from caelestia.utils.notify import close_notification, notify
from caelestia.utils.paths import recording_notif_path, recording_path, recordings_dir, user_config_path

RECORDER = "gpu-screen-recorder"


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.pause:
            subprocess.run(["pkill", "-USR2", "-f", RECORDER], stdout=subprocess.DEVNULL)
        elif self.proc_running():
            self.stop()
        else:
            self.start()

    def proc_running(self) -> bool:
        return subprocess.run(["pidof", RECORDER], stdout=subprocess.DEVNULL).returncode == 0

    def intersects(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        return a[0] < b[0] + b[2] and a[0] + a[2] > b[0] and a[1] < b[1] + b[3] and a[1] + a[3] > b[1]

    def _convert_to_physical_pixels(self, log_x: int, log_y: int, log_w: int, log_h: int, monitors: list) -> str:
        """Convert logical coordinates to physical pixels for gpu-screen-recorder.
        
        This handles fractional scaling by:
        1. Finding the minimum physical origin across all monitors
        2. Converting logical coordinates to physical coordinates using monitor scale
        """
        # Find minimum physical origin (top-left across all monitors)
        min_phys_x = 0
        min_phys_y = 0
        have_any = False
        
        for monitor in monitors:
            scale = monitor.get("scale", 1.0)
            phys_x = monitor["x"] * scale
            phys_y = monitor["y"] * scale
            
            if not have_any:
                min_phys_x = phys_x
                min_phys_y = phys_y  
                have_any = True
            else:
                min_phys_x = min(min_phys_x, phys_x)
                min_phys_y = min(min_phys_y, phys_y)
        
        # Find the monitor containing this region to get its scale
        region_monitor = None
        for monitor in monitors:
            mon_x, mon_y = monitor["x"], monitor["y"] 
            mon_w, mon_h = monitor["width"], monitor["height"]
            
            # Check if region intersects with this monitor
            if self.intersects((log_x, log_y, log_w, log_h), (mon_x, mon_y, mon_w, mon_h)):
                region_monitor = monitor
                break
        
        # Use scale from the monitor containing the region, fallback to 1.0
        scale = region_monitor.get("scale", 1.0) if region_monitor else 1.0
        
        # Convert to physical coordinates
        phys_x = max(0, round(log_x * scale - min_phys_x))
        phys_y = max(0, round(log_y * scale - min_phys_y))
        phys_w = max(1, round(log_w * scale))
        phys_h = max(1, round(log_h * scale))
        
        return f"{phys_w}x{phys_h}+{phys_x}+{phys_y}"

    def start(self) -> None:
        args = ["-w"]

        monitors = json.loads(subprocess.check_output(["hyprctl", "monitors", "-j"]))
        if self.args.region:
            if self.args.region == "slurp":
                region = subprocess.check_output(["slurp", "-f", "%wx%h+%x+%y"], text=True)
            else:
                region = self.args.region.strip()
            
            # Parse region coordinates (logical pixels from area picker)
            m = re.match(r"(\d+)x(\d+)\+(\d+)\+(\d+)", region)
            if not m:
                raise ValueError(f"Invalid region: {region}")

            log_w, log_h, log_x, log_y = map(int, m.groups())
            
            # Convert logical coordinates to physical pixels for gpu-screen-recorder
            # This handles fractional scaling correctly
            phys_region = self._convert_to_physical_pixels(log_x, log_y, log_w, log_h, monitors)
            args += ["region", "-region", phys_region]

            # Find refresh rate for the region
            r = log_x, log_y, log_w, log_h
            max_rr = 0
            for monitor in monitors:
                if self.intersects((monitor["x"], monitor["y"], monitor["width"], monitor["height"]), r):
                    rr = round(monitor["refreshRate"])
                    max_rr = max(max_rr, rr)
            args += ["-f", str(max_rr)]
        else:
            focused_monitor = next(monitor for monitor in monitors if monitor["focused"])
            if focused_monitor:
                args += [focused_monitor["name"], "-f", str(round(focused_monitor["refreshRate"]))]

        if self.args.sound:
            args += ["-a", "default_output"]

        try:
            config = json.loads(user_config_path.read_text())
            if "record" in config and "extraArgs" in config["record"]:
                args += config["record"]["extraArgs"]
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        except TypeError as e:
            raise ValueError(f"Config option 'record.extraArgs' should be an array: {e}")

        recording_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen([RECORDER, *args, "-o", str(recording_path)], start_new_session=True)

        notif = notify("-p", "Recording started", "Recording...")
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
        while self.proc_running():
            time.sleep(0.1)

        # Move to recordings folder
        new_path = recordings_dir / f"recording_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.mp4"
        recordings_dir.mkdir(exist_ok=True, parents=True)
        shutil.move(recording_path, new_path)

        # Close start notification
        try:
            close_notification(recording_notif_path.read_text())
        except IOError:
            pass

        action = notify(
            "-t", "0",  # No timeout, no close button
            "--action=play=Play",
            "--action=open=Open",
            "--action=delete=Delete",
            "Recording stopped",
            f"Recording saved in {new_path}",
        )

        if action == "play":
            subprocess.Popen(["app2unit", "-O", new_path], start_new_session=True)
        elif action == "open":
            p = subprocess.run(
                [
                    "dbus-send",
                    "--session",
                    "--dest=org.freedesktop.FileManager1",
                    "--type=method_call",
                    "/org/freedesktop/FileManager1",
                    "org.freedesktop.FileManager1.ShowItems",
                    f"array:string:file://{new_path}",
                    "string:",
                ]
            )
            if p.returncode != 0:
                subprocess.Popen(["app2unit", "-O", new_path.parent], start_new_session=True)
        elif action == "delete":
            new_path.unlink()

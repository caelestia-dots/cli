import re
import shutil
import subprocess
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from caelestia.utils import hypr
from caelestia.utils.notify import close_notification, notify
from caelestia.utils.paths import get_config, recording_notif_path, recording_path, recordings_dir

RECORDER = "gpu-screen-recorder"

# gpu-screen-recorder resolves these to whatever the current PipeWire defaults are
AUDIO_DEVICES = {
    "mic": "default_input",
    "system": "default_output",
    "combined": "default_output|default_input",
}


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.pause:
            subprocess.run(["pkill", "-USR2", "-f", RECORDER], stdout=subprocess.DEVNULL)
        elif self.args.stop:
            if self.proc_running():
                self.stop()
        elif self.proc_running():
            self.stop()
        else:
            self.start()

    def proc_running(self) -> bool:
        return subprocess.run(["pidof", RECORDER], stdout=subprocess.DEVNULL).returncode == 0

    def intersects(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        return a[0] < b[0] + b[2] and a[0] + a[2] > b[0] and a[1] < b[1] + b[3] and a[1] + a[3] > b[1]

    def slurp(self, choices: str | None = None) -> str | None:
        # Returns None when the user cancels, which is not an error
        proc = subprocess.run(["slurp", "-f", "%wx%h+%x+%y"], input=choices, capture_output=True, text=True)
        return proc.stdout.strip() if proc.returncode == 0 else None

    # Feeding slurp the window rects turns a freehand drag into a click-to-pick
    def select_window(self) -> str | None:
        windows = [c for c in hypr.message("clients") if c["mapped"] and not c["hidden"]]
        if not windows:
            raise ValueError("No windows to record")
        return self.slurp("\n".join(f"{c['at'][0]},{c['at'][1]} {c['size'][0]}x{c['size'][1]}" for c in windows))

    # Recording below the panel's rate wastes frames, above it gains nothing, so
    # match the fastest monitor the region covers
    def region_args(self, region: str) -> list[str]:
        m = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", region)
        if not m:
            raise ValueError(f"Invalid region: {region}")

        w, h, x, y = map(int, m.groups())
        r = x, y, w, h
        max_rr = 0
        for monitor in hypr.message("monitors"):
            if self.intersects((monitor["x"], monitor["y"], monitor["width"], monitor["height"]), r):
                max_rr = max(max_rr, round(monitor["refreshRate"]))

        return ["region", "-region", region, "-f", str(max_rr)]

    def start(self) -> None:
        args = ["-w"]

        if self.args.mode == "window":
            region = self.select_window()
            if region is None:
                return
            args += self.region_args(region)
        elif self.args.mode == "region" or self.args.region:
            if self.args.region and self.args.region != "slurp":
                region = self.args.region.strip()
            else:
                region = self.slurp()
                if region is None:
                    return
            args += self.region_args(region)
        else:
            monitors = hypr.message("monitors")
            focused_monitor = next(monitor for monitor in monitors if monitor["focused"])
            if focused_monitor:
                args += [focused_monitor["name"], "-f", str(round(focused_monitor["refreshRate"]))]

        device = AUDIO_DEVICES.get(self.args.audio) or ("default_output" if self.args.sound else None)
        if device:
            args += ["-a", device]

        config = get_config()
        try:
            if "record" in config and "extraArgs" in config["record"]:
                args += config["record"]["extraArgs"]
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

        if self.args.clipboard:
            file_uri = Path(new_path).resolve().as_uri() + "\n"
            subprocess.run(["wl-copy", "--type", "text/uri-list"], input=file_uri.encode())

        action = notify(
            "--action=watch=Watch",
            "--action=open=Open",
            "--action=delete=Delete",
            "Recording stopped",
            f"Recording saved in {new_path}",
        )

        if action == "watch":
            subprocess.Popen(["xdg-open", new_path], start_new_session=True)
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
                subprocess.Popen(["xdg-open", new_path.parent], start_new_session=True)
        elif action == "delete":
            new_path.unlink()

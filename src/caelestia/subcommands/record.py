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

# Handles the "Recording stopped" notification actions in a tiny detached shell
# process, so `caelestia record` can exit as soon as the recording is saved.
# $1 = saved recording path
STOPPED_NOTIF_HANDLER = r"""
action=$(notify-send -a caelestia-cli \
    --action=watch=Watch --action=open=Open --action=delete=Delete \
    "Recording stopped" "Recording saved in $1")

case "$action" in
    watch)
        xdg-open "$1" &
        ;;
    open)
        if ! dbus-send --session --dest=org.freedesktop.FileManager1 \
            --type=method_call /org/freedesktop/FileManager1 \
            org.freedesktop.FileManager1.ShowItems \
            "array:string:file://$1" "string:"; then
            xdg-open "$(dirname -- "$1")" &
        fi
        ;;
    delete)
        rm -f -- "$1"
        ;;
esac
"""


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

    def start(self) -> None:
        args = ["-w"]

        monitors = hypr.message("monitors")
        if self.args.region:
            if self.args.region == "slurp":
                region = subprocess.check_output(
                    ["slurp", "-f", "%wx%h+%x+%y"],
                    text=True,
                    stdin=subprocess.DEVNULL,
                )
            else:
                region = self.args.region.strip()
            args += ["region", "-region", region]

            m = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", region)
            if not m:
                raise ValueError(f"Invalid region: {region}")

            w, h, x, y = map(int, m.groups())
            r = x, y, w, h
            max_rr = 0
            for monitor in monitors:
                if self.intersects((monitor["x"], monitor["y"], monitor["width"], monitor["height"]), r):
                    rr = round(monitor["refreshRate"])
                    max_rr = max(max_rr, rr)
            args += ["-f", str(max_rr)]
        else:
            focused_monitor = next(monitor for monitor in monitors if monitor["focused"])
            if focused_monitor:
                args += [
                    focused_monitor["name"],
                    "-f",
                    str(round(focused_monitor["refreshRate"])),
                ]

        if self.args.sound:
            args += ["-a", "default_output"]

        config = get_config()
        try:
            if "record" in config and "extraArgs" in config["record"]:
                args += config["record"]["extraArgs"]
        except TypeError as e:
            raise ValueError(f"Config option 'record.extraArgs' should be an array: {e}")

        recording_path.parent.mkdir(parents=True, exist_ok=True)
        # The recorder outlives this command, so it must not inherit our stdio
        proc = subprocess.Popen( [RECORDER, *args, "-o", str(recording_path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

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
            # wl-copy forks a background server that outlives this command
            subprocess.run(
                ["wl-copy", "--type", "text/uri-list"],
                input=file_uri.encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # The action notification's lifetime is the user's interaction with it,
        # not this command's, so hand it off to a detached lightweight handler
        subprocess.Popen( ["sh", "-c", STOPPED_NOTIF_HANDLER, "sh", str(new_path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

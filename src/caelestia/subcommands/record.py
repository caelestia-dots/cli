import subprocess
import time
from argparse import Namespace
from datetime import datetime

from caelestia.utils.notify import notify
from caelestia.utils.paths import recording_notif_path, recording_path, recordings_dir


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        proc = subprocess.run(["pidof", "wl-screenrec"])
        if proc.returncode == 0:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        args = []

        if self.args.region:
            if self.args.region == "slurp":
                region = subprocess.check_output(["slurp"], text=True)
            else:
                region = self.args.region
            args += ["-g", region.strip()]

        if self.args.sound:
            sources = subprocess.check_output(["pactl", "list", "short", "sources"], text=True).splitlines()
            for source in sources:
                if "RUNNING" in source:
                    args += ["--audio", "--audio-device", source.split()[1]]
                    break
            else:
                raise ValueError("No audio source found")

        recording_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            ["wl-screenrec", *args, "-f", recording_path],
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        # Send notif if proc hasn't ended after a small delay
        time.sleep(0.1)
        if proc.poll() is None:
            notif = notify("-p", "Recording started", "Recording...")
            recording_notif_path.write_text(notif)
        else:
            notify("Recording failed", f"Recording failed to start: {proc.communicate()[1]}")

    def stop(self) -> None:
        subprocess.run(["pkill", "wl-screenrec"])

        # Move to recordings folder
        new_path = recordings_dir / f"recording_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.mp4"
        recordings_dir.mkdir(exist_ok=True, parents=True)
        recording_path.rename(new_path)

        # Close start notification
        try:
            notif = recording_notif_path.read_text()
            subprocess.run(
                [
                    "gdbus",
                    "call",
                    "--session",
                    "--dest=org.freedesktop.Notifications",
                    "--object-path=/org/freedesktop/Notifications",
                    "--method=org.freedesktop.Notifications.CloseNotification",
                    notif,
                ]
            )
        except IOError:
            pass

        action = notify(
            "--action=watch=Watch",
            "--action=open=Open",
            "--action=delete=Delete",
            "Recording stopped",
            f"Recording saved in {new_path}",
        )

        if action == "watch":
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

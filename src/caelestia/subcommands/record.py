import json
import shutil
import subprocess
import time
from argparse import Namespace
from datetime import datetime

from caelestia.utils.notify import notify
from caelestia.utils.paths import recording_notif_path, recording_path, recordings_dir


class Command:
    args: Namespace
    recorder: str

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.recorder = self._detect_recorder()

    def _detect_recorder(self) -> str:
        """Detect which screen recorder to use based on GPU."""
        try:
            # Check for NVIDIA GPU
            lspci_output = subprocess.check_output(["lspci"], text=True)
            if "nvidia" in lspci_output.lower():
                # Check if wf-recorder is available
                if shutil.which("wf-recorder"):
                    return "wf-recorder"

            # Default to wl-screenrec if available
            if shutil.which("wl-screenrec"):
                return "wl-screenrec"

            # Fallback to wf-recorder if wl-screenrec is not available
            if shutil.which("wf-recorder"):
                return "wf-recorder"

            raise RuntimeError("No compatible screen recorder found")
        except subprocess.CalledProcessError:
            # If lspci fails, default to wl-screenrec
            return "wl-screenrec" if shutil.which("wl-screenrec") else "wf-recorder"

    def run(self) -> None:
        if self.proc_running():
            self.stop()
        else:
            self.start()

    def proc_running(self) -> bool:
        return subprocess.run(["pidof", self.recorder], stdout=subprocess.DEVNULL).returncode == 0

    def start(self) -> None:
        args = []

        if self.args.region:
            if self.args.region == "slurp":
                region = subprocess.check_output(["slurp"], text=True)
            else:
                region = self.args.region
            args += ["-g", region.strip()]
        else:
            monitors = json.loads(subprocess.check_output(["hyprctl", "monitors", "-j"]))
            focused_monitor = next(monitor for monitor in monitors if monitor["focused"])
            if focused_monitor:
                args += ["-o", focused_monitor["name"]]

        if self.args.sound:
            sources = subprocess.check_output(["pactl", "list", "short", "sources"], text=True).splitlines()
            for source in sources:
                if "RUNNING" in source:
                    if self.recorder == "wf-recorder":
                        args += ["-a", source.split()[1]]
                    else:
                        args += ["--audio", "--audio-device", source.split()[1]]
                    break
            else:
                raise ValueError("No audio source found")

        recording_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [self.recorder, *args, "-f", recording_path],
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
        # Start killing recording process
        subprocess.run(["pkill", self.recorder])

        # Wait for recording to finish to avoid corrupted video file
        while self.proc_running():
            time.sleep(0.1)

        # Move to recordings folder
        new_path = recordings_dir / f"recording_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.mp4"
        recordings_dir.mkdir(exist_ok=True, parents=True)
        shutil.move(recording_path, new_path)

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
                ],
                stdout=subprocess.DEVNULL,
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

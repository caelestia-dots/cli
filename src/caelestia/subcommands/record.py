import json
import shutil
import subprocess
import time
from argparse import Namespace
from datetime import datetime

from caelestia.utils.notify import close_notification, notify
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
            audio_source = None

            for source in sources:
                if "RUNNING" in source:
                    audio_source = source.split()[1]
                    break

            # Fallback to IDLE source if no RUNNING source
            if not audio_source:
                for source in sources:
                    if "IDLE" in source:
                        audio_source = source.split()[1]
                        break

            if not audio_source:
                raise ValueError("No audio source found")

            if self.recorder == "wf-recorder":
                args += [f"--audio={audio_source}"]
            else:
                args += ["--audio", "--audio-device", audio_source]

        recording_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [self.recorder, *args, "-f", recording_path],
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        notif = notify("-p", "Recording started", "Recording...")
        recording_notif_path.write_text(notif)

        for _ in range(5):
            if proc.poll() is not None:
                if proc.returncode != 0:
                    close_notification(notif)
                    notify("Recording failed", f"Recording error: {proc.communicate()[1]}")
                return
            time.sleep(0.2)

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
            close_notification(recording_notif_path.read_text())
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

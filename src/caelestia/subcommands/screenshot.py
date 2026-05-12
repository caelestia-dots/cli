import subprocess
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from caelestia.utils.notify import notify
from caelestia.utils.paths import screenshots_cache_dir, screenshots_dir

LOG_FILE = Path.home() / ".local" / "share" / "caelestia" / "screenshot.log"


def log(msg: str) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True, parents=True)
    with LOG_FILE.open("a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        log(f"run() called — args.region={self.args.region!r}")
        if self.args.region:
            self.region()
        else:
            self.fullscreen()

    def region(self) -> None:
        log(f"region() called — region={self.args.region!r}")
        if self.args.region == "slurp":
            cmd = ["qs", "-c", "caelestia", "ipc", "call", "picker", "openFreeze" if self.args.freeze else "open"]
            log(f"Firing IPC: {cmd}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            log(f"IPC returned: returncode={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}")
        else:
            self._capture_region(self.args.region)

    def _capture_region(self, region: str) -> None:
        log(f"_capture_region() called — region={region!r}")
        try:
            sc_data = subprocess.check_output(
                ["grim", "-l", "0", "-g", region.strip(), "-"],
                stderr=subprocess.PIPE,
            )
            log(f"grim succeeded — {len(sc_data)} bytes captured")
        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode().strip()
            log(f"grim CalledProcessError: {msg}")
            notify("Screenshot failed", f"grim error: {msg}")
            return
        except FileNotFoundError:
            log("grim not found")
            notify("Screenshot failed", "grim not found")
            return

        try:
            swappy = subprocess.Popen(
                ["swappy", "-f", "-"],
                stdin=subprocess.PIPE,
                start_new_session=True,
                stderr=subprocess.PIPE,
            )
            log(f"swappy launched — pid={swappy.pid}")
            if swappy.stdin:
                swappy.stdin.write(sc_data)
                swappy.stdin.close()

            time.sleep(0.2)
            poll = swappy.poll()
            if poll is not None:
                _, err = swappy.communicate()
                msg = err.decode().strip()
                log(f"swappy exited early with code {poll}: {msg}")
                notify("Screenshot failed", f"swappy exited early: {msg}")
            else:
                log("swappy running OK")
        except FileNotFoundError:
            log("swappy not found")
            notify("Screenshot failed", "swappy not found")

    def fullscreen(self) -> None:
        log("fullscreen() called")
        try:
            sc_data = subprocess.check_output(["grim", "-"], stderr=subprocess.PIPE)
            log(f"grim succeeded — {len(sc_data)} bytes")
        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode().strip()
            log(f"grim CalledProcessError: {msg}")
            notify("Screenshot failed", f"grim error: {msg}")
            return
        except FileNotFoundError:
            log("grim not found")
            notify("Screenshot failed", "grim not found")
            return

        subprocess.run(["wl-copy"], input=sc_data)
        dest = screenshots_cache_dir / datetime.now().strftime("%Y%m%d%H%M%S")
        screenshots_cache_dir.mkdir(exist_ok=True, parents=True)
        dest.write_bytes(sc_data)

        action = notify(
            "-i",
            "image-x-generic-symbolic",
            "-h",
            f"STRING:image-path:{dest}",
            "--action=open=Open",
            "--action=save=Save",
            "Screenshot taken",
            f"Screenshot stored in {dest} and copied to clipboard",
        )
        log(f"notify action: {action!r}")

        if action == "open":
            subprocess.Popen(["swappy", "-f", dest], start_new_session=True)
        elif action == "save":
            new_dest = (screenshots_dir / dest.name).with_suffix(".png")
            new_dest.parent.mkdir(exist_ok=True, parents=True)
            dest.rename(new_dest)
            notify("Screenshot saved", f"Saved to {new_dest}")

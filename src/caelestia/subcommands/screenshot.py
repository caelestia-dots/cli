import subprocess
import tempfile
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from caelestia.utils.notify import notify
from caelestia.utils.paths import screenshots_cache_dir, screenshots_dir


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.region:
            self.region()
        else:
            self.fullscreen()

    def region(self) -> None:
        if self.args.clipboard:
            geometry = self.args.region
            if geometry == "slurp":
                try:
                    geometry = subprocess.check_output(["slurp", "-d"], text=True).strip()
                except subprocess.CalledProcessError:
                    return

            if not geometry:
                return

            self.capture_region_with_clipboard(geometry.strip())
            return

        if self.args.region == "slurp":
            subprocess.run(
                ["qs", "-c", "caelestia", "ipc", "call", "picker", "openFreeze" if self.args.freeze else "open"]
            )
        else:
            sc_data = subprocess.check_output(["grim", "-l", "0", "-g", self.args.region.strip(), "-"])
            swappy = subprocess.Popen(["swappy", "-f", "-"], stdin=subprocess.PIPE, start_new_session=True)
            swappy.stdin.write(sc_data)
            swappy.stdin.close()

    def fullscreen(self) -> None:
        sc_data = subprocess.check_output(["grim", "-"])

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

        if action == "open":
            subprocess.Popen(["swappy", "-f", dest], start_new_session=True)
        elif action == "save":
            new_dest = (screenshots_dir / dest.name).with_suffix(".png")
            new_dest.parent.mkdir(exist_ok=True, parents=True)
            dest.rename(new_dest)
            notify("Screenshot saved", f"Saved to {new_dest}")

    def capture_region_with_clipboard(self, geometry: str) -> None:
        tmpfile = Path(tempfile.mkstemp(prefix="caelestia-screenshot-", suffix=".png")[1])

        try:
            # Give slurp a moment to dismiss its overlay to avoid tinting the capture
            time.sleep(0.05)
            subprocess.run(["grim", "-g", geometry, str(tmpfile)], check=True)

            data = tmpfile.read_bytes()
            subprocess.run(["wl-copy", "--type", "image/png"], input=data)

            subprocess.run(["swappy", "-f", str(tmpfile), "-o", str(tmpfile)], start_new_session=True)

            data = tmpfile.read_bytes()
            subprocess.run(["wl-copy", "--type", "image/png"], input=data)
        finally:
            tmpfile.unlink(missing_ok=True)

import subprocess
from argparse import Namespace
from datetime import datetime

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

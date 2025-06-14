import subprocess
import time
from argparse import Namespace
from datetime import datetime

from caelestia.utils import hypr
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
            freeze_proc = None

            if self.args.freeze:
                freeze_proc = subprocess.Popen(["wayfreeze", "--hide-cursor"])

            try:
                ws = hypr.message("activeworkspace")["id"]
                geoms = [
                    f"{','.join(map(str, c['at']))} {'x'.join(map(str, c['size']))}"
                    for c in hypr.message("clients")
                    if c["workspace"]["id"] == ws
                ]

                # Delay to ensure wayfreeze starts first
                if freeze_proc:
                    time.sleep(0.01)

                region = subprocess.check_output(["slurp"], input="\n".join(geoms), text=True)
            finally:
                if freeze_proc:
                    freeze_proc.kill()
        else:
            region = self.args.region

        sc_data = subprocess.check_output(["grim", "-l", "0", "-g", region.strip(), "-"])
        swappy = subprocess.Popen(["swappy", "-f", "-"], stdin=subprocess.PIPE, start_new_session=True)
        swappy.stdin.write(sc_data)
        swappy.stdin.close()

    def fullscreen(self) -> None:
        sc_data = subprocess.check_output(["grim", "-"])

        subprocess.run(["wl-copy"], input=sc_data)

        dest = screenshots_cache_dir / datetime.now().strftime("%Y%m%d%H%M%S")
        screenshots_cache_dir.mkdir(exist_ok=True, parents=True)
        dest.write_bytes(sc_data)

        action = subprocess.check_output(
            [
                "notify-send",
                "-i",
                "image-x-generic-symbolic",
                "-h",
                f"STRING:image-path:{dest}",
                "-a",
                "caelestia-cli",
                "--action=open=Open",
                "--action=save=Save",
                "Screenshot taken",
                f"Screenshot stored in {dest} and copied to clipboard",
            ],
            text=True,
        ).strip()

        if action == "open":
            subprocess.Popen(["swappy", "-f", dest], start_new_session=True)
        elif action == "save":
            new_dest = (screenshots_dir / dest.name).with_suffix(".png")
            new_dest.parent.mkdir(exist_ok=True, parents=True)
            dest.rename(new_dest)
            subprocess.run(["notify-send", "Screenshot saved", f"Saved to {new_dest}"])

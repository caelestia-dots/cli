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

    def _convert_geometry_to_grim_format(self, geometry: str) -> str:
        """Convert X11 geometry format (WIDTHxHEIGHT+X+Y) to grim format (X,Y WIDTHxHEIGHT)"""
        import re
        # Match X11 geometry format: WIDTHxHEIGHT+X+Y
        match = re.match(r'(\d+)x(\d+)\+(\d+)\+(\d+)', geometry)
        if match:
            width, height, x, y = match.groups()
            return f"{x},{y} {width}x{height}"
        else:
            # If it doesn't match X11 format, assume it's already in grim format or invalid
            return geometry

    def region(self) -> None:
        if self.args.region == "slurp":
            subprocess.run(
                ["qs", "-c", "caelestia", "ipc", "call", "picker", "openFreeze" if self.args.freeze else "open"]
            )
        else:
            grim_geometry = self._convert_geometry_to_grim_format(self.args.region.strip())
            sc_data = subprocess.check_output(["grim", "-l", "0", "-g", grim_geometry, "-"])
            
            # Copy to clipboard
            subprocess.run(["wl-copy"], input=sc_data)

            # Save directly to screenshots directory with proper naming
            dest = screenshots_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.png"
            screenshots_dir.mkdir(exist_ok=True, parents=True)
            dest.write_bytes(sc_data)

            # Show notification with actions
            action = notify(
                "-t", "0",  # No timeout, no close button
                "-i",
                "image-x-generic-symbolic",
                "-h",
                f"STRING:image-path:{dest}",
                "--action=edit=Edit",
                "--action=open=Open",
                "--action=delete=Delete",
                "Screenshot taken",
                f"Screenshot saved to {dest.name} and copied to clipboard",
            )

            if action == "edit":
                subprocess.Popen(["swappy", "-f", dest], start_new_session=True)
            elif action == "open":
                p = subprocess.run(
                    [
                        "dbus-send",
                        "--session",
                        "--dest=org.freedesktop.FileManager1",
                        "--type=method_call",
                        "/org/freedesktop/FileManager1",
                        "org.freedesktop.FileManager1.ShowItems",
                        f"array:string:file://{dest}",
                        "string:",
                    ]
                )
                if p.returncode != 0:
                    subprocess.Popen(["app2unit", "-O", dest.parent], start_new_session=True)
            elif action == "delete":
                dest.unlink()
                notify("Screenshot deleted", f"Deleted {dest.name}")

    def fullscreen(self) -> None:
        sc_data = subprocess.check_output(["grim", "-"])

        subprocess.run(["wl-copy"], input=sc_data)

        # Save directly to screenshots directory with proper naming
        dest = screenshots_dir / f"screenshot_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.png"
        screenshots_dir.mkdir(exist_ok=True, parents=True)
        dest.write_bytes(sc_data)

        action = notify(
            "-t", "0",  # No timeout, no close button
            "-i",
            "image-x-generic-symbolic",
            "-h",
            f"STRING:image-path:{dest}",
            "--action=edit=Edit",
            "--action=open=Open",
            "--action=delete=Delete",
            "Screenshot taken",
            f"Screenshot saved to {dest.name} and copied to clipboard",
        )

        if action == "edit":
            subprocess.Popen(["swappy", "-f", dest], start_new_session=True)
        elif action == "open":
            p = subprocess.run(
                [
                    "dbus-send",
                    "--session",
                    "--dest=org.freedesktop.FileManager1",
                    "--type=method_call",
                    "/org/freedesktop/FileManager1",
                    "org.freedesktop.FileManager1.ShowItems",
                    f"array:string:file://{dest}",
                    "string:",
                ]
            )
            if p.returncode != 0:
                subprocess.Popen(["app2unit", "-O", dest.parent], start_new_session=True)
        elif action == "delete":
            dest.unlink()
            notify("Screenshot deleted", f"Deleted {dest.name}")

import re
import socket
from argparse import Namespace

from caelestia.utils import hypr


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.daemon:
            self.daemon()
        else:
            win = hypr.message("activewindow")
            if win["floating"]:
                self.handle_window(win["address"], win["workspace"]["name"])

    def handle_window(self, address: str, ws: str) -> None:
        mon_id = next(w for w in hypr.message("workspaces") if w["name"] == ws)["monitorID"]
        mon = next(m for m in hypr.message("monitors") if m["id"] == mon_id)
        width, height = next(c for c in hypr.message("clients") if c["address"] == address)["size"]

        scale_factor = mon["height"] / 4 / height
        scaled_win_size = f"{int(width * scale_factor)} {int(height * scale_factor)}"
        off = min(mon["width"], mon["height"]) * 0.03
        move_to = f"{int(mon['x']) + int(mon['width'] - off - width * scale_factor)} {int(mon['y']) + int(mon['height'] - off - height * scale_factor)}"

        hypr.batch(
            f"dispatch resizewindowpixel exact {scaled_win_size},address:{address}",
            f"dispatch movewindowpixel exact {move_to},address:{address}",
        )

    def daemon(self) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(hypr.socket2_path)

            while True:
                data = sock.recv(4096).decode()
                if data.startswith("openwindow>>"):
                    address, ws, cls, title = data[12:].split(",")
                    if re.match(r"^[Pp]icture(-| )in(-| )[Pp]icture$", title):
                        self.handle_window(f"0x{address}", ws)

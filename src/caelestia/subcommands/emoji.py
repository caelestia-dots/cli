import subprocess
from argparse import Namespace

from caelestia.utils.paths import cli_data_dir


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        emojis = (cli_data_dir / "emojis.txt").read_text()
        chosen = subprocess.check_output(
            ["fuzzel", "--dmenu", "--placeholder=Type to search emojis"], input=emojis, text=True
        )
        subprocess.run(["wl-copy"], input=chosen.split()[0], text=True)

import subprocess
from argparse import Namespace

from caelestia.utils.deprecation import deprecated_in_favour_of_global


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        deprecated_in_favour_of_global("clipboard", "clipboard")

        clip = subprocess.check_output(["cliphist", "list"])

        if self.args.delete:
            args = ["--prompt=del > ", "--placeholder=Delete from clipboard"]
        else:
            args = ["--placeholder=Type to search clipboard"]

        chosen = subprocess.check_output(["fuzzel", "--dmenu", *args], input=clip)

        if self.args.delete:
            subprocess.run(["cliphist", "delete"], input=chosen)
        else:
            decoded = subprocess.check_output(["cliphist", "decode"], input=chosen)
            subprocess.run(["wl-copy"], input=decoded)

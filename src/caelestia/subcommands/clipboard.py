import shutil
import subprocess
from argparse import Namespace


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        use_clipvault = shutil.which("clipvault") is not None

        if use_clipvault:
            clip = subprocess.check_output(["clipvault", "list"])
        else:
            clip = subprocess.check_output(["cliphist", "list"])

        if self.args.delete:
            args = ["--prompt=del > ", "--placeholder=Delete from clipboard"]
        else:
            args = ["--placeholder=Type to search clipboard"]

        chosen = subprocess.check_output(["fuzzel", "--dmenu", *args], input=clip)

        if self.args.delete:
            if use_clipvault:
                subprocess.run(["clipvault", "delete"], input=chosen)
            else:
                subprocess.run(["cliphist", "delete"], input=chosen)
        else:
            if use_clipvault:
                subprocess.run(["wl-copy"], input=chosen)
            else:
                decoded = subprocess.check_output(["cliphist", "decode"], input=chosen)
                subprocess.run(["wl-copy"], input=decoded)


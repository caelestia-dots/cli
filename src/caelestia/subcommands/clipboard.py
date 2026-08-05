import subprocess
from argparse import Namespace


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.paste_latest:
            latest = subprocess.check_output(["cliphist", "list"]).splitlines()[0]

            decoded = subprocess.check_output(
                ["cliphist", "decode"],
                input=latest,
            )

            text = decoded.decode("utf-8").rstrip("\n")

            subprocess.run(
                ["ydotool", "type", "-e", "0", "-d", "1", text],
                check=True,
            )
            return

        clip = subprocess.check_output(["cliphist", "list"])

        if self.args.delete:
            args = ["--prompt=del > ", "--placeholder=Delete from clipboard"]
        else:
            args = ["--placeholder=Type to search clipboard"]

        chosen = subprocess.check_output(["fuzzel", "--dmenu", *args], input=clip)

        if self.args.delete:
            subprocess.run(["cliphist", "delete"], input=chosen, check=True)
        else:
            decoded = subprocess.check_output(["cliphist", "decode"], input=chosen)
            subprocess.run(["wl-copy"], input=decoded, check=True)

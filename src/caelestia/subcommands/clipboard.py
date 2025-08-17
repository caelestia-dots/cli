#!/usr/bin/env python3
import subprocess
from argparse import ArgumentParser, Namespace

class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        clip = subprocess.check_output(["cliphist", "list"])

        if self.args.reset:
            entries = clip.splitlines()
            for entry in entries:
                subprocess.run(["cliphist", "delete"], input=entry)
            print(f"Deleted {len(entries)} clipboard entries.")
            return

        if self.args.delete:
            fuzzel_args = ["--prompt=del > ", "--placeholder=Delete from clipboard"]
        else:
            fuzzel_args = ["--placeholder=Type to search clipboard"]

        chosen = subprocess.check_output(
            ["fuzzel", "--dmenu", *fuzzel_args], input=clip
        )

        if self.args.delete:
            subprocess.run(["cliphist", "delete"], input=chosen)
        else:
            decoded = subprocess.check_output(["cliphist", "decode"], input=chosen)
            subprocess.run(["wl-copy"], input=decoded)

def main():
    parser = ArgumentParser(description="Caelestia clipboard manager")
    parser.add_argument("-d", "--delete", action="store_true", help="Delete selected clipboard entry")
    parser.add_argument("-r", "--reset", action="store_true", help="Delete all clipboard entries")
    args = parser.parse_args()

    Command(args).run()

if __name__ == "__main__":
    main()
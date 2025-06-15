import subprocess
from argparse import Namespace


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.show:
            # Print the ipc
            self.print_ipc()
        elif self.args.log:
            # Print the log
            self.print_log()
        elif self.args.message:
            # Send a message
            self.message(*self.args.message)
        else:
            # Start the shell
            subprocess.Popen(["qs", "-c", "caelestia"]).wait()

    def shell(self, *args: list[str]) -> str:
        return subprocess.check_output(["qs", "-c", "caelestia", *args], text=True)

    def print_ipc(self) -> None:
        print(self.shell("ipc", "show"), end="")

    def print_log(self) -> None:
        log = self.shell("log")
        # FIXME: remove when logging rules are added/warning is removed
        for line in log.splitlines():
            if "QProcess: Destroyed while process" not in line:
                print(line)

    def message(self, *args: list[str]) -> None:
        print(self.shell("ipc", "call", *args), end="")

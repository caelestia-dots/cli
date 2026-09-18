import json
import os
import signal
import subprocess
import time
from argparse import Namespace

from caelestia.utils.io import fatal, warn
from caelestia.utils.paths import c_cache_dir


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
        elif self.args.kill:
            # Kill the shell
            self.stop_instances()
        elif self.args.restart:
            # Restart the shell. Wait for the old instances to exit first,
            # otherwise `-n` will silently skip the relaunch
            self.stop_instances()
            self.start_shell()
        elif self.args.message:
            # Send a message
            self.message(*self.args.message)
        else:
            # Start the shell
            self.start_shell()

    def shell(self, *args: str) -> str:
        return subprocess.check_output(["qs", "-c", "caelestia", *args], text=True)

    def start_shell(self) -> None:
        args = ["qs", "-c", "caelestia", "-n"]
        if self.args.log_rules:
            args.extend(["--log-rules", self.args.log_rules])
        if self.args.daemon:
            args.append("-d")
            subprocess.run(args)
        else:
            shell = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)

            # Ensure stdout is not None for the type checker
            if shell.stdout:
                for line in shell.stdout:
                    if self.filter_log(line):
                        print(line, end="")

    def list_instances(self) -> list[dict]:
        proc = subprocess.run(["qs", "-c", "caelestia", "list", "-j"], check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            fatal(f"failed to list shell instances: {(proc.stderr or proc.stdout).strip()}")

        # `qs list` exits 0 and prints a plain text notice instead of JSON when
        # there are no instances, so only treat that exact case as empty
        out = proc.stdout.strip()
        if out.startswith("No running instances"):
            return []

        try:
            return json.loads(out)
        except json.JSONDecodeError:
            fatal(f"failed to parse shell instance list: {out}")

    def instance_pids(self) -> list[int]:
        return [instance["pid"] for instance in self.list_instances()]

    def wait_for_exit(self, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not self.list_instances():
                return True
            time.sleep(0.1)
        return False

    def stop_instances(self) -> None:
        pids = self.instance_pids()
        if not pids:
            return

        # Without `--pid`, `qs kill` only kills one instance, so ask each one to
        # exit. A kill that fails is not fatal here: the force kill below is the
        # fallback for any instance that does not go away.
        for pid in pids:
            subprocess.run(["qs", "-c", "caelestia", "kill", "--pid", str(pid)], check=False, stdout=subprocess.DEVNULL)

        # Teardown is not instant, so wait for the instances to actually disappear
        if self.wait_for_exit(5):
            return

        # Some instances are stuck; force kill them to ensure they are stopped
        warn("shell did not exit gracefully, killing")
        for pid in self.instance_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        if not self.wait_for_exit(2):
            fatal("an instance of the shell is still running")

    def filter_log(self, line: str) -> bool:
        return f"Cannot open: file://{c_cache_dir}/imagecache/" not in line

    def print_ipc(self) -> None:
        print(self.shell("ipc", "show"), end="")

    def print_log(self) -> None:
        if self.args.log_rules:
            log = self.shell("log", "-r", self.args.log_rules)
        else:
            log = self.shell("log")
        # FIXME: remove when logging rules are added/warning is removed
        for line in log.splitlines():
            if self.filter_log(line):
                print(line)

    def message(self, *args: list[str]) -> None:
        print(self.shell("ipc", "call", *args), end="")

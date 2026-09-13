import os
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if not self.args.delete and self._picker_available():
            picker = self._picker_path()
            result = subprocess.run([str(picker)], check=False)
            if result.returncode != 127:
                return

        self._run_fuzzel()

    @staticmethod
    def _picker_path() -> Path:
        override = os.environ.get("CAELESTIA_CLIPBOARD_PICKER")
        if override:
            return Path(override).expanduser()

        return Path(__file__).resolve().parent.parent / "data" / "clipboard-picker" / "picker"

    @classmethod
    def _picker_available(cls) -> bool:
        return cls._picker_path().is_file() and (shutil.which("quickshell") or shutil.which("qs")) is not None

    def _run_fuzzel(self) -> None:
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

import subprocess


def notify(*args: list[str]) -> str:
    return subprocess.check_output(["notify-send", "-a", "caelestia-cli", *args], text=True).strip()

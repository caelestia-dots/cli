import subprocess
import sys

from caelestia.utils.io import WARNING_COLOUR, format_msg
from caelestia.utils.notify import notify


def deprecated(command: str, details: list[str], notification_body: str | None = None) -> None:
    """Warn that `caelestia <command>` is deprecated. The command should still run afterwards.

    `details` are the extra lines printed under the headline (e.g. what to use instead).
    The warning is printed to stderr (so stdout stays clean for piping) and sent as a
    notification, since deprecated commands are usually run from keybinds where no
    terminal is visible. `notification_body` defaults to `details` joined by newlines.
    """

    headline = f"`caelestia {command}` is deprecated and will be removed in a future release."
    print(format_msg(WARNING_COLOUR, True, "Warning: " + "\n".join([headline, *details])), file=sys.stderr)

    try:
        notify(
            "-u",
            "normal",
            f"`caelestia {command}` is deprecated",
            notification_body if notification_body is not None else "\n".join(details),
        )
    except (OSError, subprocess.CalledProcessError):
        pass  # notify-send missing or no notification daemon, the stderr warning is enough


def deprecated_in_favour_of_global(command: str, name: str) -> None:
    """Warn that `caelestia <command>` is deprecated in favour of the `caelestia:<name>` Hyprland global dispatcher."""

    dsp = f'hl.dsp.global("caelestia:{name}")'
    cmd = f"hyprctl dispatch '{dsp}'"

    deprecated(
        command,
        [
            "Use the following in your Hyprland config instead:",
            f"  {dsp}",
            "or from the command line:",
            f"  {cmd}",
        ],
        notification_body=f"Use {dsp} instead.\nCommand line: {cmd}",
    )

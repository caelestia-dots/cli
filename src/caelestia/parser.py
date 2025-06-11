import argparse

from caelestia.subcommands import (
    clipboard,
    emoji,
    pip,
    record,
    scheme,
    screenshot,
    shell,
    toggle,
    variant,
    wallpaper,
    wsaction,
)
from caelestia.utils.scheme import get_scheme_names, scheme_variants


def parse_args() -> (argparse.ArgumentParser, argparse.Namespace):
    parser = argparse.ArgumentParser(prog="caelestia", description="Main control script for the Caelestia dotfiles")

    # Add subcommand parsers
    command_parser = parser.add_subparsers(
        title="subcommands", description="valid subcommands", metavar="COMMAND", help="the subcommand to run"
    )

    # Create parser for shell opts
    shell_parser = command_parser.add_parser("shell", help="start or message the shell")
    shell_parser.set_defaults(cls=shell.Command)
    shell_parser.add_argument("message", nargs="*", help="a message to send to the shell")
    shell_parser.add_argument("-s", "--show", action="store_true", help="print all shell IPC commands")
    shell_parser.add_argument(
        "-l",
        "--log",
        nargs="?",
        const="quickshell.dbus.properties.warning=false;quickshell.dbus.dbusmenu.warning=false;quickshell.service.notifications.warning=false;quickshell.service.sni.host.warning=false",
        metavar="RULES",
        help="print the shell log",
    )

    # Create parser for toggle opts
    toggle_parser = command_parser.add_parser("toggle", help="toggle a special workspace")
    toggle_parser.set_defaults(cls=toggle.Command)
    toggle_parser.add_argument(
        "workspace", choices=["communication", "music", "sysmon", "specialws", "todo"], help="the workspace to toggle"
    )

    # Create parser for workspace-action opts
    ws_action_parser = command_parser.add_parser(
        "workspace-action", help="execute a Hyprland workspace dispatcher in the current group"
    )
    ws_action_parser.set_defaults(cls=wsaction.Command)
    ws_action_parser.add_argument(
        "-g", "--group", action="store_true", help="whether to execute the dispatcher on a group"
    )
    ws_action_parser.add_argument(
        "dispatcher", choices=["workspace", "movetoworkspace"], help="the dispatcher to execute"
    )
    ws_action_parser.add_argument("workspace", type=int, help="the workspace to pass to the dispatcher")

    # Create parser for scheme opts
    scheme_parser = command_parser.add_parser("scheme", help="manage the colour scheme")
    scheme_parser.set_defaults(cls=scheme.Command)
    scheme_parser.add_argument("-r", "--random", action="store_true", help="switch to a random scheme")
    scheme_parser.add_argument("-n", "--name", choices=get_scheme_names(), help="the name of the scheme to switch to")
    scheme_parser.add_argument("-f", "--flavour", help="the flavour to switch to")
    scheme_parser.add_argument("-m", "--mode", choices=["dark", "light"], help="the mode to switch to")

    # Create parser for variant opts
    variant_parser = command_parser.add_parser("variant", help="manage the dynamic scheme variant")
    variant_parser.set_defaults(cls=variant.Command)
    variant_parser.add_argument("-g", "--get", action="store_true", help="print the current dynamic scheme variant")
    variant_parser.add_argument("-s", "--set", choices=scheme_variants, help="set the current dynamic scheme variant")
    variant_parser.add_argument("-r", "--random", action="store_true", help="switch to a random variant")

    # Create parser for screenshot opts
    screenshot_parser = command_parser.add_parser("screenshot", help="take a screenshot")
    screenshot_parser.set_defaults(cls=screenshot.Command)
    screenshot_parser.add_argument("-r", "--region", help="take a screenshot of a region")
    screenshot_parser.add_argument(
        "-f", "--freeze", action="store_true", help="freeze the screen while selecting a region"
    )

    # Create parser for record opts
    record_parser = command_parser.add_parser("record", help="start a screen recording")
    record_parser.set_defaults(cls=record.Command)
    record_parser.add_argument("-r", "--region", action="store_true", help="record a region")
    record_parser.add_argument("-s", "--sound", action="store_true", help="record audio")

    # Create parser for clipboard opts
    clipboard_parser = command_parser.add_parser("clipboard", help="open clipboard history")
    clipboard_parser.set_defaults(cls=clipboard.Command)
    clipboard_parser.add_argument("-d", "--delete", action="store_true", help="delete from clipboard history")

    # Create parser for emoji-picker opts
    emoji_parser = command_parser.add_parser("emoji-picker", help="toggle the emoji picker")
    emoji_parser.set_defaults(cls=emoji.Command)

    # Create parser for wallpaper opts
    wallpaper_parser = command_parser.add_parser("wallpaper", help="manage the wallpaper")
    wallpaper_parser.set_defaults(cls=wallpaper.Command)
    wallpaper_parser.add_argument("-g", "--get", action="store_true", help="print the current wallpaper")
    wallpaper_parser.add_argument("-r", "--random", action="store_true", help="switch to a random wallpaper")
    wallpaper_parser.add_argument("-f", "--file", help="the path to the wallpaper to switch to")
    wallpaper_parser.add_argument("-n", "--no-filter", action="store_true", help="do not filter by size")
    wallpaper_parser.add_argument(
        "-t",
        "--threshold",
        default=80,
        help="the minimum percentage of the largest monitor size the image must be greater than to be selected",
    )
    wallpaper_parser.add_argument(
        "-N",
        "--no-smart",
        action="store_true",
        help="do not automatically change the scheme mode based on wallpaper colour",
    )

    # Create parser for pip opts
    pip_parser = command_parser.add_parser("pip", help="picture in picture utilities")
    pip_parser.set_defaults(cls=pip.Command)
    pip_parser.add_argument("-d", "--daemon", action="store_true", help="start the daemon")

    return parser, parser.parse_args()

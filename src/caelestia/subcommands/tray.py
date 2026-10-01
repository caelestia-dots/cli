from argparse import Namespace

import dbus

from caelestia.utils.io import fatal, info, log, warn
from caelestia.utils.paths import get_shell_config


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        config = get_shell_config()
        if not isinstance(config, dict):
            fatal("Shell config must be an object")

        bar_cfg = config.get("bar", {})
        if not isinstance(bar_cfg, dict):
            fatal("Shell config option 'bar' must be an object")

        tray_cfg = bar_cfg.get("tray", {})
        if not isinstance(tray_cfg, dict):
            fatal("Shell config option 'bar.tray' must be an object")

        hidden_icons = tray_cfg.get("hiddenIcons", [])
        if not isinstance(hidden_icons, list) or not all(isinstance(item, str) for item in hidden_icons):
            fatal("Shell config option 'bar.tray.hiddenIcons' must be an array of strings")

        try:
            bus = dbus.SessionBus()
        except dbus.DBusException as e:
            fatal(f"Failed to connect to the session bus: {e}")

        try:
            watcher = bus.get_object("org.kde.StatusNotifierWatcher", "/StatusNotifierWatcher")
            items = watcher.Get(
                "org.kde.StatusNotifierWatcher",
                "RegisteredStatusNotifierItems",
                dbus_interface="org.freedesktop.DBus.Properties",
            )
        except dbus.DBusException as e:
            fatal(f"Failed to query the status notifier watcher: {e}")

        if not items:
            info("No tray items registered.")
            return

        for item in items:
            i = item.index("/")
            service = item[:i]
            path = item[i:]

            try:
                props = bus.get_object(service, path).GetAll(
                    "org.kde.StatusNotifierItem", dbus_interface="org.freedesktop.DBus.Properties"
                )
            except dbus.DBusException as e:
                warn(f"Failed to query tray item '{item}': {e}")
                continue

            icon_id = props["Id"]
            hidden = "Yes" if icon_id in hidden_icons else "No"

            info(f"Application: {props.get('Title') or '(Unknown)'}", False)
            log(f"Icon ID: {icon_id}", False)
            log(f"Hidden by config: {hidden}", False)
            log(f"Item status: {props.get('Status', 'Unknown')}", False)

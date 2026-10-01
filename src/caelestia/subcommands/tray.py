import dbus
from argparse import Namespace
from caelestia.utils.io import log, info
from caelestia.utils.paths import get_shell_config


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        bus = dbus.SessionBus()
        config = get_shell_config()
        hidden_icons = []

        try:
            if (
                "bar" in config
                and "tray" in config["bar"]
                and "hiddenIcons" in config["bar"]["tray"]
            ):
                hidden_icons = config["bar"]["tray"]["hiddenIcons"]
        except TypeError as e:
            raise ValueError(
                f"Config option 'bar.tray.hiddenIcons' should be an array: {e}"
            ) from e

        for r in bus.get_object(
            "org.kde.StatusNotifierWatcher", "/StatusNotifierWatcher"
        ).Get(
            "org.kde.StatusNotifierWatcher",
            "RegisteredStatusNotifierItems",
            dbus_interface="org.freedesktop.DBus.Properties",
        ):
            i = r.index("/")
            service = r[:i]
            path = r[i:]

            props = bus.get_object(service, path).GetAll(
                "org.kde.StatusNotifierItem",
                dbus_interface="org.freedesktop.DBus.Properties",
            )

            status = "Hidden" if props["Id"] in hidden_icons else "Visible"

            info(f"Application: {props['Title'] or '(Unknown)'}", False)
            log(f"Icon ID: {props['Id']}")
            log(f"Status: {status}")

import json
import shutil
import subprocess
from argparse import Namespace
from collections import ChainMap

from caelestia.utils import hypr
from caelestia.utils.paths import user_config_path


def is_subset(superset, subset):
    for key, value in subset.items():
        if key not in superset:
            return False

        if isinstance(value, dict):
            if not is_subset(superset[key], value):
                return False

        elif isinstance(value, str):
            if value not in superset[key]:
                return False

        elif isinstance(value, list):
            if not set(value) <= set(superset[key]):
                return False
        elif isinstance(value, set):
            if not value <= superset[key]:
                return False

        else:
            if not value == superset[key]:
                return False

    return True


class DeepChainMap(ChainMap):
    def __getitem__(self, key):
        values = (mapping[key] for mapping in self.maps if key in mapping)
        try:
            first = next(values)
        except StopIteration:
            return self.__missing__(key)
        if isinstance(first, dict):
            return self.__class__(first, *values)
        return first

    def __repr__(self):
        return repr(dict(self))


class Command:
    args: Namespace
    cfg: dict[str, dict[str, dict[str, any]]] | DeepChainMap
    clients: list[dict[str, any]] = None

    def __init__(self, args: Namespace) -> None:
        self.args = args

        self.cfg = {
            "communication": {
                "discord": {
                    "enable": True,
                    "match": [{"class": "discord"}],
                    "command": ["discord"],
                    "move": True,
                },
                "whatsapp": {
                    "enable": True,
                    "match": [{"class": "whatsapp"}],
                    "move": True,
                },
            },
            "music": {
                "spotify": {
                    "enable": True,
                    "match": [{"class": "Spotify"}, {"initialTitle": "Spotify"}, {"initialTitle": "Spotify Free"}],
                    "command": ["spicetify", "watch", "-s"],
                    "move": True,
                },
                "feishin": {
                    "enable": True,
                    "match": [{"class": "feishin"}],
                    "move": True,
                },
            },
            "sysmon": {
                "btop": {
                    "enable": True,
                    "match": [{"class": "btop", "title": "btop", "workspace": {"name": "special:sysmon"}}],
                    "command": ["foot", "-a", "btop", "-T", "btop", "fish", "-C", "exec btop"],
                },
            },
            "todo": {
                "todoist": {
                    "enable": True,
                    "match": [{"class": "Todoist"}],
                    "command": ["todoist"],
                    "move": True,
                },
            },
        }
        try:
            self.cfg = DeepChainMap(json.loads(user_config_path.read_text())["toggles"], self.cfg)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    def run(self) -> None:
        if self.args.workspace == "specialws":
            self.specialws()
            return

        for client in self.cfg[self.args.workspace].values():
            if "enable" in client and client["enable"]:
                self.handle_client_config(client)
        hypr.dispatch("togglespecialworkspace", self.args.workspace)

    def get_clients(self) -> list[dict[str, any]]:
        if self.clients is None:
            self.clients = hypr.message("clients")

        return self.clients

    def move_client(self, selector: callable, workspace: str) -> None:
        for client in self.get_clients():
            if selector(client) and client["workspace"]["name"] != f"special:{workspace}":
                hypr.dispatch("movetoworkspacesilent", f"special:{workspace},address:{client['address']}")

    def spawn_client(self, selector: callable, spawn: list[str]) -> None:
        if (spawn[0].endswith(".desktop") or shutil.which(spawn[0])) and not any(
            selector(client) for client in self.get_clients()
        ):
            subprocess.Popen(["app2unit", "--", *spawn], start_new_session=True)

    def handle_client_config(self, client: dict[str, any]) -> None:
        def selector(c: dict[str, any]) -> bool:
            # Each match is or, inside matches is and
            for match in client["match"]:
                if is_subset(c, match):
                    return True
            return False

        if "command" in client and client["command"]:
            self.spawn_client(selector, client["command"])
        if "move" in client and client["move"]:
            self.move_client(selector, self.args.workspace)

    def specialws(self) -> None:
        workspaces = hypr.message("workspaces")
        on_special_ws = any(ws["name"] == "special:special" for ws in workspaces)
        toggle_ws = "special"

        if not on_special_ws:
            active_ws = hypr.message("activewindow")["workspace"]["name"]
            if active_ws.startswith("special:"):
                toggle_ws = active_ws[8:]

        hypr.dispatch("togglespecialworkspace", toggle_ws)

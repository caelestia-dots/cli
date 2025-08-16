import json
import shutil
import subprocess
from argparse import Namespace

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


class Command:
    args: Namespace
    cfg: dict[str, dict[str, dict[str, any]]]
    clients: list[dict[str, any]] = None

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.cfg = {}

        try:
            self.cfg = json.loads(user_config_path.read_text())["toggles"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

    def run(self) -> None:
        workspace_name = self.args.workspace

        if workspace_name == "specialws":
            self.specialws()
            return

        # Workspace toggle mode with configuration
        workspaces = hypr.message("workspaces")
        workspace_status = next(
            (
                ws["windows"]
                for ws in workspaces
                if ws["name"] == f"special:{workspace_name}"
            ),
            0,
        )

        if workspace_status == 0 and workspace_name in self.cfg:
            # Workspace is empty, launch configured app in the workspace
            client = self.cfg[workspace_name]
            if "enable" in client and client["enable"] and "command" in client:
                # Apply permanent window rule if specified
                if "window_rule" in client:
                    hypr.dispatch("keyword", f"windowrulev2 {client['window_rule']}")

                # Launch app in the special workspace using hyprctl
                app_command = " ".join(client["command"])
                hypr.dispatch(
                    "exec", f"[workspace special:{workspace_name}] {app_command}"
                )

                # Don't toggle immediately, let the app launch and show the workspace
                return

        # Only toggle if workspace already exists with windows
        hypr.dispatch("togglespecialworkspace", workspace_name)

    def get_clients(self) -> list[dict[str, any]]:
        if self.clients is None:
            self.clients = hypr.message("clients")

        return self.clients

    def move_client(self, selector: callable, workspace: str) -> None:
        for client in self.get_clients():
            if (
                selector(client)
                and client["workspace"]["name"] != f"special:{workspace}"
            ):
                hypr.dispatch(
                    "movetoworkspacesilent",
                    f"special:{workspace},address:{client['address']}",
                )

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

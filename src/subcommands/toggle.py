from argparse import Namespace

from utils import hypr


class Command:
    args: Namespace
    clients: list[dict[str, any]] = None
    app2unit: str = None

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        getattr(self, self.args.workspace)()

    def get_clients(self) -> list[dict[str, any]]:
        if self.clients is None:
            self.clients = hypr.message("clients")

        return self.clients

    def get_app2unit(self) -> str:
        if self.app2unit is None:
            import shutil

            self.app2unit = shutil.which("app2unit")

        return self.app2unit

    def move_client(self, selector: callable, workspace: str) -> None:
        for client in self.get_clients():
            if selector(client):
                hypr.dispatch("movetoworkspacesilent", f"special:{workspace},address:{client['address']}")

    def spawn_client(self, selector: callable, spawn: list[str]) -> bool:
        exists = any(selector(client) for client in self.get_clients())

        if not exists:
            import subprocess

            subprocess.Popen([self.get_app2unit(), "--", *spawn], start_new_session=True)

        return not exists

    def spawn_or_move(self, selector: callable, spawn: list[str], workspace: str) -> None:
        if not self.spawn_client(selector, spawn):
            self.move_client(selector, workspace)

    def communication(self) -> None:
        self.spawn_or_move(lambda c: c["class"] == "discord", ["discord"], "communication")
        self.move_client(lambda c: c["class"] == "whatsapp", "communication")

    def music(self) -> None:
        self.spawn_or_move(
            lambda c: c["class"] == "Spotify" or c["initialTitle"] == "Spotify" or c["initialTitle"] == "Spotify Free",
            ["spicetify", "watch", "-s"],
            "music",
        )
        self.move_client(lambda c: c["class"] == "feishin", "music")

    def sysmon(self) -> None:
        self.spawn_client(
            lambda c: c["class"] == "btop" and c["title"] == "btop" and c["workspace"]["name"] == "special:sysmon",
            ["foot", "-a", "btop", "-T", "btop", "--", "btop"],
            "sysmon",
        )

    def todo(self) -> None:
        self.spawn_or_move(lambda c: c["class"] == "Todoist", ["todoist"], "todo")

    def specialws(self) -> None:
        workspaces = hypr.message("workspaces")
        on_special_ws = any(ws["name"] == "special:special" for ws in workspaces)
        toggle_ws = "special"

        if not on_special_ws:
            active_ws = hypr.message("activewindow")["workspace"]["name"]
            if active_ws.startswith("special:"):
                toggle_ws = active_ws[8:]

        hypr.dispatch("togglespecialworkspace", toggle_ws)

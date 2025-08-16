import json
import os
import socket
import subprocess
import time
from argparse import Namespace
from pathlib import Path

from caelestia.utils.paths import user_config_path


class WindowRule:
    def __init__(self, name: str, match_type: str, width: str, height: str, actions: list[str]):
        self.name = name
        self.match_type = match_type
        self.width = width
        self.height = height
        self.actions = actions


class Command:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.timeout_tracker: dict[str, float] = {}
        self.window_rules = self._load_window_rules()

    def _load_window_rules(self) -> list[WindowRule]:
        default_rules = [
            WindowRule("Write: (no subject)", "initial_title", "50%", "54%", ["float", "center"]),
            WindowRule("(Bitwarden", "title_contains", "20%", "54%", ["float", "center"]),
            WindowRule("Sign in - Google Accounts", "title_contains", "35%", "65%", ["float", "center"]),
            WindowRule("oauth", "title_contains", "30%", "60%", ["float", "center"]),
        ]

        try:
            config = json.loads(user_config_path.read_text())
            if "resizer" in config and "rules" in config["resizer"]:
                rules = []
                for rule_config in config["resizer"]["rules"]:
                    rules.append(
                        WindowRule(
                            rule_config["name"],
                            rule_config["match_type"],
                            rule_config["width"],
                            rule_config["height"],
                            rule_config["actions"],
                        )
                    )
                return rules
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        return default_rules

    def _log_message(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _is_rate_limited(self, key: str) -> bool:
        current_time = time.time()
        last_time = self.timeout_tracker.get(key, 0)

        if current_time < last_time + 1:
            return True

        self.timeout_tracker[key] = current_time
        return False

    def _get_window_info(self, window_id: str) -> dict | None:
        try:
            result = subprocess.run(["hyprctl", "clients", "-j"], capture_output=True, text=True, check=True)
            clients = json.loads(result.stdout)

            for client in clients:
                if client["address"] == f"0x{window_id}":
                    return client
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass

        return None

    def _apply_window_actions(self, window_id: str, width: str, height: str, actions: list[str]) -> bool:
        dispatch_commands = []

        if "float" in actions:
            window_info = self._get_window_info(window_id)
            if window_info and not window_info.get("floating", False):
                dispatch_commands.append(f"dispatch togglefloating address:0x{window_id}")

        dispatch_commands.append(f"dispatch resizewindowpixel exact {width} {height},address:0x{window_id}")

        if "center" in actions:
            dispatch_commands.append("dispatch centerwindow")

        try:
            subprocess.run(["hyprctl", "--batch", "; ".join(dispatch_commands)], check=True, capture_output=True)
            self._log_message(f"Applied actions to window 0x{window_id}: {width} x {height} ({', '.join(actions)})")
            return True
        except subprocess.CalledProcessError as e:
            self._log_message(f"ERROR: Failed to apply window actions for window 0x{window_id}: {e}")
            return False

    def _match_window_rule(self, window_title: str, initial_title: str) -> WindowRule | None:
        for rule in self.window_rules:
            if rule.match_type == "initial_title":
                if initial_title == rule.name:
                    return rule
            elif rule.match_type == "title_contains":
                if rule.name in window_title:
                    return rule
            elif rule.match_type == "title_exact":
                if window_title == rule.name:
                    return rule

        return None

    def _handle_window_event(self, event: str) -> None:
        if not event.startswith("windowtitle"):
            return

        try:
            window_id = event.split(">>")[1].split(",")[0]

            if not all(c in "0123456789abcdefABCDEF" for c in window_id):
                self._log_message(f"ERROR: Invalid window ID format: {window_id}")
                return

            window_info = self._get_window_info(window_id)
            if not window_info:
                return

            window_title = window_info.get("title", "")
            initial_title = window_info.get("initialTitle", "")

            self._log_message(f"DEBUG: Window 0x{window_id} - Title: '{window_title}' | Initial: '{initial_title}'")

            rule = self._match_window_rule(window_title, initial_title)
            if rule:
                if self._is_rate_limited(window_id):
                    self._log_message(f"Rate limited: skipping window 0x{window_id}")
                    return

                self._log_message(f"Matched rule '{rule.name}' for window 0x{window_id}")
                self._apply_window_actions(window_id, rule.width, rule.height, rule.actions)

        except (IndexError, ValueError) as e:
            self._log_message(f"ERROR: Failed to parse window event: {e}")

    def run(self) -> None:
        if self.args.daemon:
            self._run_daemon()
        else:
            print("Resizer daemon - use --daemon to start")

    def _run_daemon(self) -> None:
        self._log_message("Hyprland window resizer started")
        self._log_message(f"Loaded {len(self.window_rules)} window rules")

        xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        hyprland_signature = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")

        if not xdg_runtime_dir or not hyprland_signature:
            self._log_message("ERROR: Required environment variables not set")
            return

        socket_path = Path(xdg_runtime_dir) / "hypr" / hyprland_signature / ".socket2.sock"
        if not socket_path.exists():
            self._log_message(f"ERROR: Hyprland socket not found at {socket_path}")
            return

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(str(socket_path))
                sock_file = sock.makefile("r")

                self._log_message("Connected to Hyprland socket, listening for events...")

                for line in sock_file:
                    line = line.strip()
                    if line:
                        self._handle_window_event(line)

        except KeyboardInterrupt:
            self._log_message("Resizer daemon stopped")
        except Exception as e:
            self._log_message(f"ERROR: {e}")

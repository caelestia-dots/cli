import json
import os
import re
import socket
import subprocess
import time
from argparse import Namespace
from pathlib import Path

from caelestia.utils import hypr
from caelestia.utils.config_watcher import setup_config_watcher
from caelestia.utils.paths import user_config_path


class WindowRule:
    def __init__(self, name: str, match_type: str, width: str, height: str, actions: list[str], padding: int = 20):
        self.name = name
        self.match_type = match_type
        self.width = width
        self.height = height
        self.actions = actions
        self.padding = padding


class Command:
    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.timeout_tracker: dict[str, float] = {}
        self.window_rules = self._load_window_rules()
        self.observer = None
        self._setup_file_watcher()

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
                            rule_config.get("padding", 20),
                        )
                    )
                return rules
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        return default_rules

    def _setup_file_watcher(self) -> None:
        """Set up file watching for config changes"""
        self.observer = setup_config_watcher(self)

    def _reload_rules(self) -> None:
        """Reload window rules from config file"""
        try:
            new_rules = self._load_window_rules()
            self.window_rules = new_rules
            self._log_message(f"Reloaded {len(self.window_rules)} window rules")
        except Exception as e:
            self._log_message(f"ERROR: Failed to reload rules: {e}")

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

    def _apply_window_actions(
        self, window_id: str, width: str, height: str, actions: list[str], padding: int = 20
    ) -> bool:
        success = True

        if "float" in actions:
            window_info = self._get_window_info(window_id)
            if window_info and not window_info.get("floating", False):
                success = success and hypr.dispatch("togglefloating", f"address:0x{window_id}")

        if "pip" in actions:
            success = success and hypr.dispatch("pin", f"address:0x{window_id}")
        else:
            if width and height:
                success = success and hypr.dispatch(
                    "resizewindowpixel", f"exact {width} {height},address:0x{window_id}"
                )

        if "center" in actions:
            success = success and hypr.dispatch("centerwindow")

        # Position window in corners using pixel coordinates
        if any(corner in actions for corner in ["bottom_right", "bottom_left", "top_right", "top_left"]):
            try:
                # Get monitor info to calculate corner positions
                monitor_info = hypr.message("monitors")[0]  # Use first/primary monitor
                monitor_width = monitor_info["width"]
                monitor_height = monitor_info["height"]

                # Parse width and height to get actual pixel values
                if width.endswith("%"):
                    actual_width = int(monitor_width * int(width[:-1]) / 100)
                else:
                    actual_width = int(width)

                if height.endswith("%"):
                    actual_height = int(monitor_height * int(height[:-1]) / 100)
                else:
                    actual_height = int(height)

                # Calculate position based on corner
                if "bottom_right" in actions:
                    x = monitor_width - actual_width - padding
                    y = monitor_height - actual_height - padding
                elif "bottom_left" in actions:
                    x = padding
                    y = monitor_height - actual_height - padding
                elif "top_right" in actions:
                    x = monitor_width - actual_width - padding
                    y = padding
                elif "top_left" in actions:
                    x = padding
                    y = padding

                success = success and hypr.dispatch("movewindowpixel", f"exact {x} {y},address:0x{window_id}")
            except (KeyError, ValueError, IndexError):
                # Fallback to center if corner positioning fails
                success = success and hypr.dispatch("centerwindow")

        if success:
            if width and height:
                self._log_message(f"Applied actions to window 0x{window_id}: {width} x {height} ({', '.join(actions)})")
            else:
                self._log_message(f"Applied actions to window 0x{window_id}: {', '.join(actions)}")
            return True
        else:
            self._log_message(f"ERROR: Failed to apply window actions for window 0x{window_id}")
            return False

    def _match_window_rule(self, window_title: str, initial_title: str) -> WindowRule | None:
        self._log_message(f"DEBUG: Checking {len(self.window_rules)} rules for title '{window_title}'")
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
            elif rule.match_type == "title_regex":
                try:
                    if re.search(rule.name, window_title):
                        return rule
                except re.error as e:
                    self._log_message(f"ERROR: Invalid regex pattern '{rule.name}': {e}")

        return None

    def _handle_window_event(self, event: str) -> None:
        if not (event.startswith("windowtitle") or event.startswith("openwindow")):
            return

        try:
            if event.startswith("openwindow"):
                window_id = event.split(">>")[1].split(",")[0]
            else:  # windowtitle
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
                self._apply_window_actions(window_id, rule.width, rule.height, rule.actions, rule.padding)

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

        # Start file watcher
        if self.observer:
            self.observer.start()
            self._log_message("Config file watcher started")

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
        finally:
            # Stop file watcher
            if self.observer:
                self.observer.stop()
                self.observer.join()
                self._log_message("Config file watcher stopped")

import json
import re
import socket
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

from caelestia.utils import hypr
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
            WindowRule("(Bitwarden", "titleContains", "20%", "54%", ["float", "center"]),
            WindowRule("Sign in - Google Accounts", "titleContains", "35%", "65%", ["float", "center"]),
            WindowRule("oauth", "titleContains", "30%", "60%", ["float", "center"]),
            WindowRule("^[Pp]icture(-| )in(-| )[Pp]icture$", "titleRegex", "", "", ["pip"]),
        ]

        try:
            config = json.loads(user_config_path.read_text())
            if "resizer" in config and "rules" in config["resizer"]:
                rules = []
                for rule_config in config["resizer"]["rules"]:
                    rules.append(
                        WindowRule(
                            rule_config["name"],
                            rule_config["matchType"],
                            rule_config["width"],
                            rule_config["height"],
                            rule_config["actions"],
                        )
                    )
                return rules
        except (json.JSONDecodeError, KeyError):
            self._log_message("ERROR: invalid config")
        except FileNotFoundError:
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

    def _get_window_info(self, window_id: str) -> Optional[Dict[str, Any]]:
        try:
            clients = hypr.message("clients")
            if isinstance(clients, list):
                for client in clients:
                    if isinstance(client, dict) and client.get("address") == f"0x{window_id}":
                        return client
        except Exception:
            pass

        return None

    def _apply_pip_action(self, window_id: str) -> None:
        try:
            address = f"0x{window_id}"
            clients_result = hypr.message("clients")
            if not isinstance(clients_result, list):
                return

            window = None
            for c in clients_result:
                if isinstance(c, dict) and c.get("address") == address:
                    window = c
                    break

            if not window or not isinstance(window, dict) or not window.get("floating", False):
                return

            workspaces_result = hypr.message("workspaces")
            if not isinstance(workspaces_result, list):
                return

            workspace_info = window.get("workspace")
            if not isinstance(workspace_info, dict):
                return

            workspace_name = workspace_info.get("name")
            workspace = None
            for w in workspaces_result:
                if isinstance(w, dict) and w.get("name") == workspace_name:
                    workspace = w
                    break

            if not workspace or not isinstance(workspace, dict):
                return

            monitors_result = hypr.message("monitors")
            if not isinstance(monitors_result, list):
                return

            monitor_id = workspace.get("monitorID")
            monitor = None
            for m in monitors_result:
                if isinstance(m, dict) and m.get("id") == monitor_id:
                    monitor = m
                    break

            if not monitor or not isinstance(monitor, dict):
                return

            window_size = window.get("size")
            if not isinstance(window_size, list) or len(window_size) < 2:
                return

            width, height = window_size[0], window_size[1]
            if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
                return

            monitor_height = monitor.get("height")
            monitor_width = monitor.get("width")
            monitor_scale = monitor.get("scale")
            monitor_x = monitor.get("x")
            monitor_y = monitor.get("y")

            if not all(isinstance(x, (int, float)) for x in [monitor_height, monitor_width, monitor_scale, monitor_x, monitor_y]):
                return
            
            monitor_height = monitor_height / monitor_scale
            monitor_width = monitor_width / monitor_scale

            scale_factor = monitor_height / 4 / height
            scaled_width = int(width * scale_factor)
            scaled_height = int(height * scale_factor)

            # Ensure minimum reasonable size
            min_width = 200
            min_height = 150
            scaled_width = max(scaled_width, min_width)
            scaled_height = max(scaled_height, min_height)

            # Use offset to ensure window stays on screen with some margin
            offset = min(monitor_width, monitor_height) * 0.03

            # Position in bottom-right corner with offset
            move_x = monitor_x + monitor_width - scaled_width - offset
            move_y = monitor_y + monitor_height - scaled_height - offset

            command1 = f"dispatch resizewindowpixel exact {scaled_width} {scaled_height},address:{address}"
            command2 = f"dispatch movewindowpixel exact {int(move_x)} {int(move_y)},address:{address}"
            hypr.batch(command1, command2)

            self._log_message(
                f"Applied PiP action to window {address}: {scaled_width}x{scaled_height} at ({move_x}, {move_y})"
            )

        except Exception as e:
            self._log_message(f"ERROR: Failed to apply PiP action to window 0x{window_id}: {e}")

    def _apply_window_actions(self, window_id: str, width: str, height: str, actions: list[str]) -> bool:
        dispatch_commands = []

        if "float" in actions:
            window_info = self._get_window_info(window_id)
            if window_info and not window_info.get("floating", False):
                dispatch_commands.append(f"dispatch togglefloating address:0x{window_id}")

        if "pip" in actions:
            self._apply_pip_action(window_id)
            return True

        dispatch_commands.append(f"dispatch resizewindowpixel exact {width} {height},address:0x{window_id}")

        if "center" in actions:
            dispatch_commands.append("dispatch centerwindow")

        try:
            hypr.batch(*dispatch_commands)
            self._log_message(f"Applied actions to window 0x{window_id}: {width} x {height} ({', '.join(actions)})")
            return True
        except Exception as e:
            self._log_message(f"ERROR: Failed to apply window actions for window 0x{window_id}: {e}")
            return False

    def _match_window_rule(self, window_title: str, initial_title: str) -> WindowRule | None:
        for rule in self.window_rules:
            if rule.match_type == "initialTitle":
                if initial_title == rule.name:
                    return rule
            elif rule.match_type == "titleContains":
                if rule.name in window_title:
                    return rule
            elif rule.match_type == "titleExact":
                if window_title == rule.name:
                    return rule
            elif rule.match_type == "titleRegex":
                try:
                    if re.search(rule.name, window_title):
                        return rule
                except re.error:
                    self._log_message(f"ERROR: Invalid regex pattern in rule '{rule.name}'")

        return None

    def _handle_window_event(self, event: str) -> None:
        if event.startswith("windowtitle"):
            self._handle_title_event(event)
        elif event.startswith("openwindow"):
            self._handle_open_event(event)

    def _handle_title_event(self, event: str) -> None:
        try:
            # Handle both >> and >>> separators (different Hyprland versions)
            if ">>>" in event:
                window_id = event.split(">>>")[1].split(",")[0]
            else:
                window_id = event.split(">>")[1].split(",")[0]
            
            # Remove any leading > characters
            window_id = window_id.lstrip(">")

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
            self._log_message(f"ERROR: Failed to parse window title event: {e}")

    def _handle_open_event(self, event: str) -> None:
        try:
            # Handle both >> and >>> separators
            if "openwindow>>>" in event:
                data = event[13:]  # Remove "openwindow>>>"
            else:
                data = event[12:]  # Remove "openwindow>>"
            
            window_id, workspace, window_class, title = data.split(",", 3)
            
            # Remove any leading > characters
            window_id = window_id.lstrip(">")

            if not all(c in "0123456789abcdefABCDEF" for c in window_id):
                self._log_message(f"ERROR: Invalid window ID format: {window_id}")
                return

            self._log_message(f"DEBUG: New window 0x{window_id} - Title: '{title}' | Class: '{window_class}'")

            rule = self._match_window_rule(title, title)
            if rule:
                if self._is_rate_limited(window_id):
                    self._log_message(f"Rate limited: skipping window 0x{window_id}")
                    return

                self._log_message(f"Matched rule '{rule.name}' for new window 0x{window_id}")
                self._apply_window_actions(window_id, rule.width, rule.height, rule.actions)

        except (IndexError, ValueError) as e:
            self._log_message(f"ERROR: Failed to parse window open event: {e}")

    def run(self) -> None:
        if self.args.daemon:
            self._run_daemon()
        elif hasattr(self.args, "pattern") and self.args.pattern == "pip":
            self._run_pip_mode()
        elif all(
            hasattr(self.args, attr) and getattr(self.args, attr)
            for attr in ["pattern", "match_type", "width", "height", "actions"]
        ):
            self._run_active_mode()
        else:
            print(
                "Resizer daemon - use --daemon to start, 'pip' for quick pip mode, or provide pattern, match_type, width, height, and actions for active mode"
            )

    def _run_pip_mode(self) -> None:
        """Quick pip mode - applies pip action to the active window if it's floating"""
        try:
            active_window_result = hypr.message("activewindow")
            if not isinstance(active_window_result, dict) or not active_window_result.get("address"):
                print("ERROR: No active window found")
                return

            address = active_window_result.get("address", "")
            if not isinstance(address, str) or not address.startswith("0x"):
                print("ERROR: Invalid window address")
                return

            window_id = address[2:]  # Remove "0x" prefix
            window_title = active_window_result.get("title", "")

            if not active_window_result.get("floating", False):
                print(f"Window '{window_title}' is not floating. PIP only works on floating windows.")
                print("Try making it floating first with: hyprctl dispatch togglefloating")
                return

            print(f"Applying PIP to active window: '{window_title}'")
            self._apply_pip_action(window_id)
            print("PIP applied successfully")

        except Exception as e:
            print(f"ERROR: Failed to apply PIP to active window: {e}")

    def _run_active_mode(self) -> None:
        try:
            # Create a temporary rule from command line arguments
            actions = self.args.actions.split(",") if self.args.actions else []
            temp_rule = WindowRule(self.args.pattern, self.args.match_type, self.args.width, self.args.height, actions)

            # Special case: "active" pattern means only target the currently active window
            if temp_rule.name.lower() == "active":
                self._apply_to_active_window(temp_rule)
                return

            # Find all windows that match the pattern
            matching_windows = self._find_matching_windows(temp_rule)
            
            if not matching_windows:
                print(f"No windows found matching pattern '{temp_rule.name}' with match type '{temp_rule.match_type}'")
                return

            print(f"Found {len(matching_windows)} matching window(s)")
            
            # Apply rule to all matching windows
            success_count = 0
            for window in matching_windows:
                window_id = window["address"][2:]  # Remove "0x" prefix
                window_title = window.get("title", "")
                
                print(f"Applying rule to window 0x{window_id}: '{window_title}'")
                success = self._apply_window_actions(window_id, temp_rule.width, temp_rule.height, temp_rule.actions)
                if success:
                    success_count += 1

            print(f"Successfully applied rule to {success_count}/{len(matching_windows)} windows")

        except Exception as e:
            print(f"ERROR: Failed to apply rule: {e}")

    def _apply_to_active_window(self, temp_rule: WindowRule) -> None:
        """Apply rule only to the currently active window"""
        try:
            active_window_result = hypr.message("activewindow")
            if not isinstance(active_window_result, dict) or not active_window_result.get("address"):
                print("ERROR: No active window found")
                return

            window_title = active_window_result.get("title", "")
            address = active_window_result.get("address", "")
            if not isinstance(address, str) or not address.startswith("0x"):
                print("ERROR: Invalid window address")
                return

            window_id = address[2:]  # Remove "0x" prefix
            
            print(f"Applying rule to active window 0x{window_id}: '{window_title}'")
            success = self._apply_window_actions(window_id, temp_rule.width, temp_rule.height, temp_rule.actions)
            if success:
                print("Rule applied successfully")
            else:
                print("Failed to apply rule")

        except Exception as e:
            print(f"ERROR: Failed to apply rule to active window: {e}")

    def _find_matching_windows(self, temp_rule: WindowRule) -> list:
        """Find all windows that match the given rule pattern"""
        try:
            clients_result = hypr.message("clients")
            if not isinstance(clients_result, list):
                return []

            matching_windows = []
            for window in clients_result:
                if not isinstance(window, dict):
                    continue

                window_title = window.get("title", "")
                initial_title = window.get("initialTitle", "")
                
                # Check if window matches the pattern
                matches = False
                if temp_rule.match_type == "initialTitle":
                    matches = initial_title == temp_rule.name
                elif temp_rule.match_type == "titleContains":
                    matches = temp_rule.name in window_title
                elif temp_rule.match_type == "titleExact":
                    matches = window_title == temp_rule.name
                elif temp_rule.match_type == "titleRegex":
                    try:
                        matches = bool(re.search(temp_rule.name, window_title))
                    except re.error:
                        print(f"ERROR: Invalid regex pattern '{temp_rule.name}'")
                        return []

                if matches:
                    matching_windows.append(window)

            return matching_windows

        except Exception as e:
            print(f"ERROR: Failed to find matching windows: {e}")
            return []

    def _run_daemon(self) -> None:
        self._log_message("Hyprland window resizer started")
        self._log_message(f"Loaded {len(self.window_rules)} window rules")

        socket_path = Path(hypr.socket2_path)
        if not socket_path.exists():
            self._log_message(f"ERROR: Hyprland socket not found at {socket_path}")
            return

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.connect(hypr.socket2_path)

                self._log_message("Connected to Hyprland socket, listening for events...")

                while True:
                    data = sock.recv(4096).decode()
                    if data:
                        for line in data.strip().split("\n"):
                            if line:
                                self._handle_window_event(line)

        except KeyboardInterrupt:
            self._log_message("Resizer daemon stopped")
        except Exception as e:
            self._log_message(f"ERROR: {e}")

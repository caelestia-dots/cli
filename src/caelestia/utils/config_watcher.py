import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from caelestia.utils.paths import user_config_path


class ConfigWatcher(FileSystemEventHandler):
    def __init__(self, command_instance):
        self.command_instance = command_instance
        self._last_reload_time = 0
        self._debounce_delay = 0.5  # 500ms debounce to avoid duplicate reloads

    def on_modified(self, event):
        if event.is_directory:
            return
        # Debug logging
        self.command_instance._log_message(f"DEBUG: File event: {event.src_path}")
        if event.src_path == str(user_config_path):
            current_time = time.time()
            # Debounce multiple file system events
            if current_time - self._last_reload_time < self._debounce_delay:
                return

            self._last_reload_time = current_time
            self.command_instance._log_message("Config file changed, reloading rules...")
            self.command_instance._reload_rules()


def setup_config_watcher(command_instance) -> Observer | None:
    """Set up file watching for config changes"""
    if user_config_path.exists():
        command_instance._log_message(f"DEBUG: Watching directory: {user_config_path.parent}")
        command_instance._log_message(f"DEBUG: Config file path: {user_config_path}")
        observer = Observer()
        event_handler = ConfigWatcher(command_instance)
        observer.schedule(event_handler, str(user_config_path.parent), recursive=False)
        return observer
    return None

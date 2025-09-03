import json
import os
import subprocess
import time
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from caelestia.utils.notify import notify


class Command:
    args: Namespace
    config_dir: Path
    timer_file: Path

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.config_dir = Path.home() / ".config" / "caelestia"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.timer_file = self.config_dir / "timers.json"

    def run(self) -> None:
        if self.args.list:
            self.list_timers()
        elif self.args.show is not None:
            self.show_timer(self.args.show)
        elif self.args.quit is not None:
            self.quit_timer(self.args.quit)
        elif self.args.duration:
            self.start_timer(self.args.duration)
        else:
            print("Usage: caelestia timer <duration> | --list | --show <id> | --quit <id>")
            print("Set a timer for <duration> minutes, or use options to manage existing timers.")

    def load_timers(self) -> Dict:
        """Load timers from JSON file, cleaning up finished ones."""
        if not self.timer_file.exists():
            return {"timers": []}
        
        try:
            with open(self.timer_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"timers": []}
        
        # Clean up finished timers
        active_timers = []
        for timer in data.get("timers", []):
            try:
                if self._is_process_running(timer["pid"]):
                    active_timers.append(timer)
            except (KeyError, ValueError):
                continue
        
        data["timers"] = active_timers
        self.save_timers(data)
        return data

    def save_timers(self, data: Dict) -> None:
        """Save timers to JSON file."""
        with open(self.timer_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def start_timer(self, duration: str) -> None:
        """Start a new timer."""
        try:
            minutes = float(duration)
            if minutes <= 0:
                print("Error: Please provide a positive number for minutes (e.g., 60 or 0.5).")
                return
        except ValueError:
            print("Error: Please provide a valid number for minutes (e.g., 60 or 0.5).")
            return

        seconds = int(minutes * 60)
        start_time = datetime.now().timestamp()
        
        # Create background command exactly matching original shell script
        cmd = (
            f"sleep {seconds} && "
            f'notify-send -u low -i dialog-information-symbolic "Timer" "{minutes} minute timer done!" && '
            f"paplay /usr/share/sounds/freedesktop/stereo/complete.oga"
        )
        
        # Start background process
        process = subprocess.Popen(["sh", "-c", cmd])
        
        # Load existing timers and add new one
        data = self.load_timers()
        timer_data = {
            "pid": process.pid,
            "start_time": start_time,
            "duration": seconds,
            "minutes": minutes
        }
        data["timers"].append(timer_data)
        self.save_timers(data)
        
        print(f"Timer set for {minutes} minutes ({seconds} seconds). Notification and sound will trigger when done.")

    def list_timers(self) -> None:
        """List all running timers."""
        data = self.load_timers()
        timers = data.get("timers", [])
        
        if not timers:
            print("No timers are currently running.")
            return
        
        for i, timer in enumerate(timers):
            minutes = timer.get("minutes", timer["duration"] / 60)
            print(f"Timer {i}: Set for {minutes:.2f} minutes")

    def show_timer(self, index: int) -> None:
        """Show remaining time for a specific timer."""
        data = self.load_timers()
        timers = data.get("timers", [])
        
        if index < 0 or index >= len(timers):
            print(f"Error: No timer found at index {index} or timer has already finished.")
            return
        
        timer = timers[index]
        current_time = datetime.now().timestamp()
        elapsed = current_time - timer["start_time"]
        remaining = timer["duration"] - elapsed
        
        if remaining > 0:
            minutes = remaining / 60
            print(f"Timer {index}: {minutes:.2f} minutes remaining")
        else:
            print(f"Timer {index}: Already finished")

    def quit_timer(self, index: int) -> None:
        """Quit a specific timer."""
        data = self.load_timers()
        timers = data.get("timers", [])
        
        if index < 0 or index >= len(timers):
            print(f"Error: No timer found at index {index} or timer has already finished.")
            return
        
        timer = timers[index]
        
        # Kill the process
        try:
            os.kill(timer["pid"], 9)  # SIGKILL
            print(f"Timer {index} has been terminated.")
        except OSError:
            print(f"Timer {index} was already finished.")
        
        # Remove from list
        del timers[index]
        self.save_timers(data)
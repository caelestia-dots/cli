import subprocess
import os
from argparse import Namespace


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "timer")
        
        # Build the command arguments
        cmd = [script_path]
        
        # Pass through all arguments from argparse to the script
        if hasattr(self.args, 'list') and self.args.list:
            cmd.extend(['-l'])
        elif hasattr(self.args, 'show') and self.args.show is not None:
            cmd.extend(['-s', str(self.args.show)])
        elif hasattr(self.args, 'quit') and self.args.quit is not None:
            cmd.extend(['-q', str(self.args.quit)])
        elif hasattr(self.args, 'duration') and self.args.duration is not None:
            cmd.append(str(self.args.duration))
        
        # Execute the timer script
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout, end='')
            if result.stderr:
                print(result.stderr, end='')
            exit(result.returncode)
        except Exception as e:
            print(f"Error executing timer script: {e}")
            exit(1)

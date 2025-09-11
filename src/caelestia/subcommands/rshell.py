import subprocess
import os
from argparse import Namespace


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        script_path = os.path.join(os.path.dirname(__file__), "scripts", "rshell")
        
        # Execute the rshell script
        try:
            result = subprocess.run([script_path], capture_output=True, text=True)
            if result.stdout:
                print(result.stdout, end='')
            if result.stderr:
                print(result.stderr, end='')
            exit(result.returncode)
        except Exception as e:
            print(f"Error executing rshell script: {e}")
            exit(1)
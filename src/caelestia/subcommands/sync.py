import os
import subprocess
from argparse import Namespace


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        home = os.path.expanduser("~")
        dots_base = os.path.join(home, "dots")
        repos = {
            "xiu (dotfiles)": os.path.join(dots_base, "xiu"),
            "xiu-shell": os.path.join(dots_base, "xiu-shell"),
            "xiu-cli": os.path.join(dots_base, "xiu-cli"),
        }

        print("\033[1;34m=== Synchronizing with upstream (caelestia-dots) ===\033[0m\n")

        for name, path in repos.items():
            if not os.path.exists(path) or not os.path.exists(os.path.join(path, ".git")):
                continue

            print(f"\033[1;36mSyncing {name}...\033[0m")
            try:
                print("  [1/3] Fetching upstream...")
                subprocess.run(["git", "-C", path, "fetch", "upstream"], check=True)

                print("  [2/3] Merging upstream/main into current branch...")
                res = subprocess.run(
                    ["git", "-C", path, "merge", "upstream/main", "-m", "chore: sync with upstream/main"],
                    capture_output=True,
                    text=True,
                )
                if res.returncode != 0:
                    print(f"\033[1;31m  Merge conflict encountered in {name}:\033[0m\n{res.stderr or res.stdout}")
                    continue
                else:
                    print(f"  \033[32m{res.stdout.strip()}\033[0m")

                if not getattr(self.args, "no_push", False):
                    print("  [3/3] Pushing to origin/xiu...")
                    subprocess.run(["git", "-C", path, "push", "origin", "HEAD"], check=True)
                    print("  \033[32m✓ Pushed successfully.\033[0m")

            except subprocess.CalledProcessError as e:
                print(f"\033[1;31m  Failed to sync {name}: {e}\033[0m")

            print()

        print("\033[1;32m✓ Sync workflow complete.\033[0m\n")

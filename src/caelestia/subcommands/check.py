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

        print("\033[1;34m=== xiu Upstream Drift & Health Check ===\033[0m\n")

        for name, path in repos.items():
            if not os.path.exists(path) or not os.path.exists(os.path.join(path, ".git")):
                print(f"\033[1;33m[WARN]\033[0m {name}: Repository path {path} not found")
                continue

            print(f"\033[1;36mChecking {name} ({path})...\033[0m")

            # Fetch upstream silently if possible
            try:
                subprocess.run(
                    ["git", "-C", path, "fetch", "upstream"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            except Exception:
                pass

            # Check branch
            branch = subprocess.check_output(
                ["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"], text=True
            ).strip()

            # Ahead/Behind upstream/main
            try:
                counts = subprocess.check_output(
                    ["git", "-C", path, "rev-list", "--left-right", "--count", "HEAD...upstream/main"],
                    text=True,
                ).strip().split()
                ahead = int(counts[0]) if len(counts) > 0 else 0
                behind = int(counts[1]) if len(counts) > 1 else 0
            except Exception:
                ahead = behind = 0

            print(f"  Branch: \033[1m{branch}\033[0m")
            print(f"  Ahead of upstream/main: \033[32m{ahead} commits\033[0m")
            if behind > 0:
                print(f"  Behind upstream/main:  \033[33m{behind} commits (sync recommended)\033[0m")
            else:
                print("  Behind upstream/main:  \033[32m0 commits (fully up to date)\033[0m")

            # Check local working tree cleanliness
            status = subprocess.check_output(
                ["git", "-C", path, "status", "--porcelain"], text=True
            ).strip()
            if status:
                print("  Working tree: \033[33mUncommitted changes present\033[0m")
                for line in status.splitlines()[:5]:
                    print(f"    {line}")
                if len(status.splitlines()) > 5:
                    print(f"    ... and {len(status.splitlines()) - 5} more")
            else:
                print("  Working tree: \033[32mClean\033[0m")

            # Check diff against upstream/main outside overlay paths
            try:
                diff_files = subprocess.check_output(
                    ["git", "-C", path, "diff", "--name-only", "upstream/main...HEAD"],
                    text=True,
                ).strip().splitlines()
                core_modified = [
                    f for f in diff_files if not f.startswith("modules/pill/") and not "pillconfig" in f
                ]
                print(f"  Core upstream files modified: {len(core_modified)}")
                for f in core_modified[:5]:
                    print(f"    - {f}")
            except Exception:
                pass
            print()

        print("\033[1;32m✓ Health check complete.\033[0m To sync upstream changes, run: \033[1mxiu sync\033[0m\n")

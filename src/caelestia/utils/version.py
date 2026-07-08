import shutil
import subprocess

from caelestia.utils.paths import config_dir


def fetch_git_metadata(repo_dir, branch="upstream/main") -> tuple[str, str] | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-list", "--format=%B", "--max-count=1", branch],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None

    lines = output.strip().splitlines()
    return lines[0].split()[1], "".join(lines[1:])


def print_version() -> None:
    if shutil.which("pacman"):
        print("Packages:")
        pkgs = ["caelestia-shell", "caelestia-cli", "caelestia-meta"]
        versions = subprocess.run(
            ["pacman", "-Q", *pkgs], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        ).stdout

        for pkg in pkgs:
            if pkg not in versions:
                print(f"    {pkg} not installed")
        version_lines = versions.splitlines()
        if version_lines:
            print("\n".join(f"    {pkg}" for pkg in version_lines))
    else:
        print("Packages: not on Arch")

    print()
    caelestia_dir = (config_dir / "hypr").resolve().parent
    caelestia_metadata = fetch_git_metadata(caelestia_dir, "HEAD")

    if caelestia_metadata:
        commit, message = caelestia_metadata
        print("Caelestia:")
        print("    Last commit:", commit)
        print("    Commit message:", message)
    else:
        print("Caelestia: not installed")

    print()
    try:
        shell_ver = subprocess.check_output(["/usr/lib/caelestia/version", "-s"], text=True).strip()
        print("Shell:")
        print("    ", shell_ver)
    except FileNotFoundError:
        print("Shell: version helper not available")

    print()
    if shutil.which("qs"):
        print("Quickshell:")
        print("   ", subprocess.check_output(["qs", "--version"], text=True).strip())
    else:
        print("Quickshell: not in PATH")

    local_shell_dir = config_dir / "quickshell/caelestia"
    if local_shell_dir.exists():
        print("\nLocal copy of shell found:")
        upstream_metadata = fetch_git_metadata(local_shell_dir)

        if upstream_metadata:
            commit, message = upstream_metadata
            print("    Last merged upstream commit:", commit)
            print("    Commit message:", message)
        else:
            print("    Unable to determine last merged upstream commit.")

        local_metadata = fetch_git_metadata(local_shell_dir, "HEAD")
        if local_metadata:
            commit, message = local_metadata
            print("\n    Last local commit:", commit)
            print("    Commit message:", message)

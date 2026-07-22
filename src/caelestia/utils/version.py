import shutil
import subprocess
import sys
from pathlib import Path

from caelestia.utils.dots.legacy import LEGACY_META_PKG, detect_legacy_repo
from caelestia.utils.dots.packages import ArchInstaller
from caelestia.utils.dots.source import DotsSource, SourceError
from caelestia.utils.dots.state import DotsState
from caelestia.utils.paths import config_dir

PKGS = ("caelestia-shell", "caelestia-cli")


def _header(text: str) -> None:
    if sys.stdout.isatty():
        print(f"\033[1m\033[36m{text}\033[0m")
    else:
        print(text)


def fetch_git_metadata(repo_dir: Path, branch: str = "upstream/main") -> tuple[str, str] | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", repo_dir, "show", "-s", "--format=%H%x00%s", branch],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    commit, separator, message = output.rstrip("\n").partition("\0")
    return (commit, message) if separator else None


def print_packages() -> tuple[str, str] | None:
    if not shutil.which("pacman"):
        print("Packages: not on Arch")
        return None

    _header("Packages:")
    installer = ArchInstaller("")  # Dummy helper cause we only use query
    installed = [(pkg, installer.query(pkg)) for pkg in PKGS]
    for pkg, result in installed:
        if result is None:
            print(f"    {pkg}: not installed")
    for _, result in installed:
        if result is not None:
            name, version = result
            print(f"    {name}: {version}")

    return installer.query(LEGACY_META_PKG)


def print_legacy_install(meta_package: tuple[str, str] | None) -> None:
    legacy_path = detect_legacy_repo()
    if legacy_path is None and meta_package is None:
        return

    print()
    _header("Legacy install detected:")
    print("    Legacy dots path:", legacy_path or "not found")

    if meta_package is None:
        print(f"    {LEGACY_META_PKG}: not installed")
    else:
        name, version = meta_package
        print(f"    {name}: {version}")
    print("    Please update the CLI to the latest version and run 'caelestia install' to update the dots.")


def print_dots_version() -> None:
    applied_rev = DotsState.load().applied_rev
    if applied_rev is None:
        _header("Dots:\033[0m not installed")
        return

    _header("Dots:")
    print("    Last commit:", applied_rev)
    source = DotsSource()
    try:
        message = source.commit_message_at(applied_rev)
    except (SourceError, FileNotFoundError):
        print("    Commit message: unavailable")
    else:
        print("    Commit message:", message)


def print_version() -> None:
    meta_package = print_packages()
    print_legacy_install(meta_package)

    print()
    print_dots_version()

    print()
    try:
        shell_ver = subprocess.check_output(["/usr/lib/caelestia/version", "-s"], text=True).strip()
        _header("Shell:")
        print("   ", shell_ver)
    except FileNotFoundError:
        _header("Shell:\033[0m version helper not available")

    print()
    if shutil.which("qs"):
        _header("Quickshell:")
        print("   ", subprocess.check_output(["qs", "--version"], text=True).strip())
    else:
        _header("Quickshell:\033[0m not in PATH")

    local_shell_dir = config_dir / "quickshell/caelestia"
    if local_shell_dir.exists():
        print()
        _header("Local copy of shell found:")
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

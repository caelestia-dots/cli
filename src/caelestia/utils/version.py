import shutil
import subprocess
import sys
from pathlib import Path

from caelestia.utils.dots.legacy import LEGACY_META_PKG, detect_legacy_repo
from caelestia.utils.dots.packages import ArchInstaller
from caelestia.utils.dots.source import DotsSource, SourceError
from caelestia.utils.dots.state import DotsState
from caelestia.utils.paths import config_dir

PKGS = ("caelestia-shell", "caelestia-cli", "quickshell")
INDENT = "    "


def _header(text: str, suffix: str = "") -> None:
    suffix = f" {suffix}" if suffix else ""
    if sys.stdout.isatty():
        print(f"\033[1;36m{text}\033[0m{suffix}")
    else:
        print(f"{text}{suffix}")


def _rows(pairs: list[tuple[str, str]], align: bool = False) -> None:
    dim = sys.stdout.isatty()
    width = max((len(key) for key, _ in pairs), default=0) if align else 0
    for key, value in pairs:
        if dim:
            value = f"\033[2m{value}\033[0m"
        label = key.ljust(width) if align else f"{key}:"
        print(f"{INDENT}{label}  {value}" if align else f"{INDENT}{label} {value}")


def _commit(commit: str, message: str) -> str:
    subject = message.splitlines()[0] if message else ""
    return f"{commit[:7]} ({subject})"


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
    missing = [(pkg, "not installed") for pkg, result in installed if result is None]
    present = [result for _, result in installed if result is not None]
    _rows(missing + present, align=True)

    return installer.query(LEGACY_META_PKG)


def print_legacy_install(meta_package: tuple[str, str] | None) -> None:
    legacy_path = detect_legacy_repo()
    if legacy_path is None and meta_package is None:
        return

    print()
    _header("Legacy install detected:")
    meta_row = (LEGACY_META_PKG, "not installed") if meta_package is None else meta_package
    _rows([("Legacy dots path", str(legacy_path or "not found")), meta_row])
    update_msg = "Please update the CLI to the latest version and run 'caelestia install' to update the dots."
    if sys.stdout.isatty():
        update_msg = f"\033[1m{update_msg}\033[0m"
    print(f"{INDENT}{update_msg}")


def print_dots_version() -> None:
    state = DotsState.load()
    if state.applied_rev is None:
        _header("Dots:", "not installed")
        return

    _header("Dots:")
    source = DotsSource()
    try:
        message = source.commit_message_at(state.applied_rev)
    except (SourceError, FileNotFoundError):
        message = ""
    components = ", ".join(state.enabled_components) or "none"
    _rows(
        [
            ("Commit", _commit(state.applied_rev, message)),
            ("Components", components),
            ("AUR helper", state.aur_helper),
        ]
    )


def print_version() -> None:
    meta_package = print_packages()
    print_legacy_install(meta_package)

    print()
    print_dots_version()

    print()
    try:
        shell_ver = subprocess.check_output(["/usr/lib/caelestia/version", "-s"], text=True).strip()
        _header("Shell:")
        print(f"{INDENT}{shell_ver}")
    except FileNotFoundError:
        _header("Shell:", "version helper not available")

    print()
    if shutil.which("qs"):
        qs_ver = subprocess.check_output(["qs", "--version"], text=True).strip()
        _header("Quickshell:")
        print(f"{INDENT}{qs_ver}")
    else:
        _header("Quickshell:", "not in PATH")

    local_shell_dir = config_dir / "quickshell/caelestia"
    if local_shell_dir.exists():
        print()
        _header("Local copy of shell found:")
        upstream_metadata = fetch_git_metadata(local_shell_dir)

        if upstream_metadata:
            commit, message = upstream_metadata
            _rows([("Last merged upstream commit", _commit(commit, message))])
        else:
            print(f"{INDENT}Unable to determine last merged upstream commit.")

        local_metadata = fetch_git_metadata(local_shell_dir, "HEAD")
        if local_metadata:
            commit, message = local_metadata
            _rows([("Last local commit", _commit(commit, message))])

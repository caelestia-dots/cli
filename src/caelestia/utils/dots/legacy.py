import subprocess
from pathlib import Path

from caelestia.utils.paths import config_dir, state_dir

LEGACY_META_PKG = "caelestia-meta"

_confs = [
    "hypr",
    "starship.toml",
    "foot",
    "fish",
    "fastfetch",
    "uwsm",
    "btop",
    "spicetify",
    "Code/User/settings.json",
    "VSCodium/User/settings.json",
    "Code/User/keybindings.json",
    "VSCodium/User/keybindings.json",
    "code-flags.conf",
    "codium-flags.conf",
]


def _find_legacy_repo(path: Path) -> Path | None:
    try:
        remote = subprocess.check_output(["git", "-C", path, "remote", "get-url", "origin"], text=True)
    except subprocess.CalledProcessError:
        return

    # Check remote
    if remote.strip() != "https://github.com/caelestia-dots/caelestia.git":
        return

    # Walk up parents to try to find one that is a git repo
    while path != Path.home() and not (path / ".git").is_dir():
        path = path.parent

    # Only return path if didn't hit home (we really don't want to nuke home)
    if path != Path.home():
        return path


def detect_legacy_repo() -> Path | None:
    for conf in _confs:
        path = config_dir / conf
        if not path.is_symlink():
            continue

        legacy_dir = _find_legacy_repo(path.resolve())
        if legacy_dir:
            return legacy_dir

    return _find_legacy_repo(state_dir / "caelestia")


def legacy_to_delete(legacy_dir: Path) -> list[Path]:
    to_delete = []

    for conf in _confs:
        path = config_dir / conf
        if path.is_symlink() and legacy_dir in path.resolve().parents:
            to_delete.append(path)

    others = [
        *(Path.home() / ".zen").glob("*/chrome/userChrome.css"),
        Path.home() / ".local/lib/caelestia/caelestiafox",
    ]
    for path in others:
        if path.is_symlink() and legacy_dir in path.resolve().parents:
            to_delete.append(path)

    to_delete.append(legacy_dir)

    return to_delete

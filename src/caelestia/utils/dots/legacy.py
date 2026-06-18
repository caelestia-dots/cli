import subprocess
from pathlib import Path

from caelestia.utils.paths import config_dir

LEGACY_META_PKG = "caelestia-meta"


def legacy_to_delete() -> list[Path]:
    legacy_dir = (config_dir / "hypr").resolve().parent
    if not (legacy_dir / "install.fish").is_file():
        return []

    try:
        remote = subprocess.check_output(["git", "-C", legacy_dir, "remote", "get-url", "origin"], text=True)
    except subprocess.CalledProcessError:
        return []

    if remote != "https://github.com/caelestia-dots/caelestia.git":
        return []

    to_delete = []
    confs = [
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
    for conf in confs:
        path = config_dir / conf
        if path.is_symlink() and legacy_dir in path.resolve().parents:
            to_delete.append(path)

    others = [
        (Path.home() / ".zen").glob("*/chrome/userChrome.css"),
        Path.home() / ".local/lib/caelestia/caelestiafox",
    ]
    for path in others:
        if path.is_symlink() and legacy_dir in path.resolve().parents:
            to_delete.append(path)

    to_delete.append(legacy_dir)

    return to_delete

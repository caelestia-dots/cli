import subprocess
from pathlib import Path

from caelestia.utils.paths import config_dir, data_dir

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
        remote = subprocess.check_output(
            ["git", "-C", path, "remote", "get-url", "origin"], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return

    # Check remote
    if remote.strip() != "https://github.com/caelestia-dots/caelestia.git":
        return

    # Ignore anything outside home
    if Path.home() not in path.parents:
        return

    # Walk up parents (capped at home) to find the repo root
    while path != Path.home() and not (path / ".git").is_dir():
        path = path.parent

    # Only return path if didn't hit home (we really don't want to nuke home)
    if path != Path.home():
        return path


def _filter_candidates(candidates: list[Path], legacy_dir: Path) -> list[Path]:
    return [path for path in candidates if path.is_symlink() and legacy_dir in path.resolve().parents]


def detect_legacy_repo() -> Path | None:
    for conf in _confs:
        path = config_dir / conf
        if not path.is_symlink():
            continue

        legacy_dir = _find_legacy_repo(path.resolve())
        if legacy_dir:
            return legacy_dir

    return _find_legacy_repo(data_dir / "caelestia")


def legacy_config_symlinks(base: Path, legacy_dir: Path | None) -> list[Path]:
    """Config-relative links install.fish created, resolved under `base` (the live config or a backup of it)."""

    if not legacy_dir:
        return []

    candidates = [base / conf for conf in _confs]
    return _filter_candidates(candidates, legacy_dir)


def legacy_symlinks(legacy_dir: Path | None) -> list[Path]:
    """All paths symlinked into the legacy repo (the links install.fish created)."""

    if not legacy_dir:
        return []

    extras = [
        *(Path.home() / ".zen").glob("*/chrome/userChrome.css"),
        Path.home() / ".local/lib/caelestia/caelestiafox",
    ]

    return [*legacy_config_symlinks(config_dir, legacy_dir), *_filter_candidates(extras, legacy_dir)]


def legacy_to_delete(legacy_dir: Path | None) -> list[Path]:
    if not legacy_dir:
        return []

    non_syms = [Path.home() / ".mozilla/native-messaging-hosts/caelestiafox.json"]

    return [
        *legacy_symlinks(legacy_dir),
        *(p for p in non_syms if p.exists()),
        legacy_dir,
    ]

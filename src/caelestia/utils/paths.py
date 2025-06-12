import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

config_dir = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
data_dir = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share"))
state_dir = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local/state"))
cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))

c_config_dir = config_dir / "caelestia"
c_data_dir = data_dir / "caelestia"
c_state_dir = state_dir / "caelestia"
c_cache_dir = cache_dir / "caelestia"

cli_data_dir = Path(__file__).parent.parent / "data"
templates_dir = cli_data_dir / "templates"

scheme_path = c_state_dir / "scheme.json"
scheme_data_dir = cli_data_dir / "schemes"
scheme_cache_dir = c_cache_dir / "schemes"

wallpapers_dir = Path.home() / "Pictures/Wallpapers"
wallpaper_path_path = c_state_dir / "wallpaper/path.txt"
wallpaper_link_path = c_state_dir / "wallpaper/current"
wallpaper_thumbnail_path = c_state_dir / "wallpaper/thumbnail.jpg"
wallpapers_cache_dir = c_cache_dir / "wallpapers"

screenshots_dir = Path.home() / "Pictures/Screenshots"
screenshots_cache_dir = c_cache_dir / "screenshots"


def compute_hash(path: Path | str) -> str:
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)

    return sha.hexdigest()


def atomic_dump(path: Path, content: dict[str, any]) -> None:
    with tempfile.NamedTemporaryFile("w") as f:
        json.dump(content, f)
        f.flush()
        shutil.move(f.name, path)

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
pictures_dir = Path(os.getenv("XDG_PICTURES_DIR", Path.home() / "Pictures"))
videos_dir = Path(os.getenv("XDG_VIDEOS_DIR", Path.home() / "Videos"))

c_config_dir = config_dir / "caelestia"
c_data_dir = data_dir / "caelestia"
c_state_dir = state_dir / "caelestia"
c_cache_dir = cache_dir / "caelestia"

user_config_path = c_config_dir / "cli.json"
cli_data_dir = Path(__file__).parent.parent / "data"
templates_dir = cli_data_dir / "templates"
user_templates_dir = c_config_dir / "templates"
theme_dir = c_state_dir / "theme"

scheme_path = c_state_dir / "scheme.json"
scheme_data_dir = cli_data_dir / "schemes"
scheme_cache_dir = c_cache_dir / "schemes"

wallpapers_dir = os.getenv("CAELESTIA_WALLPAPERS_DIR", pictures_dir / "Wallpapers")
wallpaper_path_path = c_state_dir / "wallpaper/path.txt"
wallpaper_link_path = c_state_dir / "wallpaper/current"
wallpaper_thumbnail_path = c_state_dir / "wallpaper/thumbnail.jpg"
wallpapers_cache_dir = c_cache_dir / "wallpapers"

screenshots_dir = os.getenv("CAELESTIA_SCREENSHOTS_DIR", pictures_dir / "Screenshots")
screenshots_cache_dir = c_cache_dir / "screenshots"

recordings_dir = os.getenv("CAELESTIA_RECORDINGS_DIR", videos_dir / "Recordings")
recording_path = c_state_dir / "record/recording.mp4"
recording_notif_path = c_state_dir / "record/notifid.txt"


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

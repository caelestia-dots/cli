import json
import os
import random
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import cast

from materialyoucolor.hct import Hct
from materialyoucolor.utils.color_utils import argb_from_rgb
from PIL import Image

from caelestia.utils.colourfulness import get_variant
from caelestia.utils.hypr import message
from caelestia.utils.material import get_colours_for_image
from caelestia.utils.paths import (
    c_cache_dir,
    compute_hash,
    get_config,
    wallpaper_link_path,
    wallpaper_path_path,
    wallpaper_thumbnail_path,
    wallpapers_cache_dir,
)
from caelestia.utils.scheme import Scheme, get_scheme
from caelestia.utils.theme import apply_colours

VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mkv"})


def is_valid_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif"]


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def is_valid_wallpaper(path: Path) -> bool:
    return is_valid_image(path) or is_video(path)


def video_thumbnail_hash(path: Path | str) -> str:
    """Return the DJB2 path hash used by the shell's QML thumbnail lookup."""
    encoded_path = str(Path(path).resolve()).encode("utf-16-le")
    value = 5381
    for offset in range(0, len(encoded_path), 2):
        code_unit = int.from_bytes(encoded_path[offset : offset + 2], "little")
        value = ((value << 5) + value + code_unit) & 0xFFFFFFFF
    return str(value)


def get_video_thumbnail_path(wall: Path | str) -> Path:
    return c_cache_dir / "videothumbs" / f"{video_thumbnail_hash(wall)}.jpg"


def _probe_video(wall: Path) -> tuple[int, int, float]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(wall),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    duration = float(data.get("format", {}).get("duration", 0))
    return int(stream["width"]), int(stream["height"]), duration


def extract_video_thumbnail(wall: Path | str) -> Path:
    wall = Path(wall).resolve()
    if not is_video(wall):
        raise ValueError(f'"{wall}" is not a valid video wallpaper')

    thumbnail = get_video_thumbnail_path(wall)
    if thumbnail.exists() and thumbnail.stat().st_mtime_ns >= wall.stat().st_mtime_ns:
        return thumbnail

    _, _, duration = _probe_video(wall)
    seek = min(max(duration * 0.3, 0), max(duration - 0.05, 0))
    temporary = thumbnail.with_name(f".{thumbnail.name}.{os.getpid()}.tmp.jpg")
    thumbnail.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(seek),
                "-i",
                str(wall),
                "-frames:v",
                "1",
                "-vf",
                "scale=256:144:force_original_aspect_ratio=increase,crop=256:144",
                "-q:v",
                "3",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(thumbnail)
    finally:
        temporary.unlink(missing_ok=True)

    return thumbnail


def extract_video_thumbnails(directory: Path | str) -> list[Path]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f'"{directory}" is not a directory')

    return [extract_video_thumbnail(wall) for wall in directory.rglob("*") if is_video(wall)]


def check_wall(wall: Path, filter_size: tuple[int, int], threshold: float) -> bool:
    if is_video(wall):
        width, height, _ = _probe_video(wall)
    else:
        with Image.open(wall) as img:
            width, height = img.size

    return width >= filter_size[0] * threshold and height >= filter_size[1] * threshold


def get_wallpaper() -> str | None:
    try:
        return wallpaper_path_path.read_text()
    except IOError:
        return None


def get_wallpapers(args: Namespace) -> list[Path]:
    directory = Path(args.random)
    if not directory.is_dir():
        return []

    walls = [f for f in directory.rglob("*") if is_valid_wallpaper(f)]

    if args.no_filter:
        return walls

    monitors = cast(list[dict[str, int]], message("monitors"))
    filter_size = min(m["width"] for m in monitors), min(m["height"] for m in monitors)

    return [f for f in walls if check_wall(f, filter_size, args.threshold)]


def get_thumb(wall: Path, cache: Path) -> Path:
    thumb = cache / "thumbnail.jpg"

    if not thumb.exists():
        with Image.open(wall) as img:
            img = img.convert("RGB")
            img.thumbnail((128, 128), Image.Resampling.NEAREST)
            thumb.parent.mkdir(parents=True, exist_ok=True)
            img.save(thumb, "JPEG")

    return thumb


def get_smart_opts(wall: Path, cache: Path) -> dict:
    opts_cache = cache / "smart.json"

    try:
        return json.loads(opts_cache.read_text())
    except (IOError, json.JSONDecodeError):
        pass

    opts = {}

    with Image.open(get_thumb(wall, cache)) as img:
        opts["variant"] = get_variant(img)
        img.thumbnail((1, 1), Image.Resampling.LANCZOS)

        # Cast the pixel to a tuple of 3 integers to safely unpack it
        pixel = cast(tuple[int, int, int], img.getpixel((0, 0)))
        hct = Hct.from_int(argb_from_rgb(*pixel))

        opts["mode"] = "light" if hct.tone > 60 else "dark"

    opts_cache.parent.mkdir(parents=True, exist_ok=True)
    with opts_cache.open("w") as f:
        json.dump(opts, f)

    return opts


def get_colours_for_wall(wall: Path | str, no_smart: bool) -> None:
    wall = Path(wall)
    scheme = get_scheme()
    cache = wallpapers_cache_dir / compute_hash(wall)

    if wall.suffix.lower() == ".gif":
        wall = convert_gif(wall)
    elif is_video(wall):
        wall = extract_video_thumbnail(wall)

    name = "dynamic"

    if not no_smart:
        smart_opts = get_smart_opts(wall, cache)
        scheme = Scheme(
            {
                "name": name,
                "flavour": scheme.flavour,
                "mode": smart_opts["mode"],
                "variant": smart_opts["variant"],
                "colours": scheme.colours,
            }
        )

    return {
        "name": name,
        "flavour": scheme.flavour,
        "mode": scheme.mode,
        "variant": scheme.variant,
        "colours": get_colours_for_image(get_thumb(wall, cache), scheme),
    }


def convert_gif(wall: Path) -> Path:
    cache = wallpapers_cache_dir / compute_hash(wall)
    output_path = cache / "first_frame.png"

    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(wall) as img:
            try:
                img.seek(0)
            except EOFError:
                pass

            img = img.convert("RGB")
            img.save(output_path, "PNG")

    return output_path


def set_wallpaper(wall: Path, no_smart: bool) -> None:
    # Make path absolute
    wall = Path(wall).resolve()

    if not is_valid_wallpaper(wall):
        raise ValueError(f'"{wall}" is not a valid wallpaper')

    # Use a still frame for animated wallpaper thumbnails and colour extraction.
    if wall.suffix.lower() == ".gif":
        wall_cache = convert_gif(wall)
    elif is_video(wall):
        wall_cache = extract_video_thumbnail(wall)
    else:
        wall_cache = wall

    # Update files
    wallpaper_path_path.parent.mkdir(parents=True, exist_ok=True)
    wallpaper_path_path.write_text(str(wall))
    wallpaper_link_path.parent.mkdir(parents=True, exist_ok=True)
    wallpaper_link_path.unlink(missing_ok=True)
    wallpaper_link_path.symlink_to(wall)

    cache = wallpapers_cache_dir / compute_hash(wall_cache)

    # Generate thumbnail or get from cache
    thumb = get_thumb(wall_cache, cache)
    wallpaper_thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    wallpaper_thumbnail_path.unlink(missing_ok=True)
    wallpaper_thumbnail_path.symlink_to(thumb)

    scheme = get_scheme()

    # Change mode and variant based on wallpaper colour
    if scheme.name == "dynamic" and not no_smart:
        smart_opts = get_smart_opts(wall_cache, cache)
        scheme.mode = smart_opts["mode"]
        scheme.variant = smart_opts["variant"]

    # Update colours
    scheme.update_colours()
    apply_colours(scheme.colours, scheme.mode)

    # Run custom post-hook if configured
    cfg = get_config().get("wallpaper", {})
    if post_hook := cfg.get("postHook"):
        subprocess.run(
            post_hook,
            shell=True,
            env={
                **os.environ,
                "WALLPAPER_PATH": str(wall),
                "SCHEME_NAME": scheme.name,
                "SCHEME_FLAVOUR": scheme.flavour,
                "SCHEME_MODE": scheme.mode,
                "SCHEME_VARIANT": scheme.variant,
                "SCHEME_COLOURS": json.dumps(scheme.colours),
                "THUMBNAIL_PATH": str(thumb),
            },
            stderr=subprocess.DEVNULL,
        )


def set_random(args: Namespace) -> None:
    wallpapers = get_wallpapers(args)

    if not wallpapers:
        raise ValueError("No valid wallpapers found")

    try:
        last_wall = wallpaper_path_path.read_text()
        wallpapers.remove(Path(last_wall))

        if not wallpapers:
            raise ValueError("Only valid wallpaper is current")
    except (FileNotFoundError, ValueError):
        pass

    set_wallpaper(random.choice(wallpapers), args.no_smart)

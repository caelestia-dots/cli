import json
import os
import random
import subprocess
from argparse import Namespace
from pathlib import Path

from materialyoucolor.hct import Hct
from materialyoucolor.utils.color_utils import argb_from_rgb
from PIL import Image, ImageOps

from caelestia.utils.hypr import message
from caelestia.utils.material import get_colours_for_image
from caelestia.utils.paths import (
    compute_hash,
    user_config_path,
    wallpaper_link_path,
    wallpaper_path_path,
    wallpaper_thumbnail_path,
    wallpapers_cache_dir,
)
from caelestia.utils.scheme import Scheme, get_scheme
from caelestia.utils.theme import apply_colours


def is_valid_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif"]


def _extract_animated_metadata(path: Path) -> dict:
    """
    Detects animated image and its metadata from 'path'
    """
    try:
        with Image.open(path) as img:
            is_animated = getattr(img, "is_animated", False)
            n_frames = getattr(img, "n_frames", 1) if is_animated else 1
            fmt = getattr(img, "format", None)

            per_frame_duration = img.info.get("duration", 0) # in ms
            loop = img.info.get("loop")

            total_duration = None
            if is_animated and per_frame_duration and n_frames:
                try:
                    total_duration = int(per_frame_duration) * int(n_frames) # in ms
                except Exception:
                    pass

            return {
                "is_animated": is_animated,
                "format": fmt,
                "n_frames": n_frames,
                "frame_duration_ms": int(per_frame_duration) \
                    if isinstance(per_frame_duration, (int, float)) else None,
                "total_duration_ms": int(total_duration) \
                    if isinstance(total_duration, (int, float)) else None,
                "loop": loop if isinstance(loop, int) else None,
            }

    except Exception:
        return {
            "is_animated": False,
            "format": None,
            "n_frames": 1,
            "frame_duration_ms": None,
            "total_duration_ms": None,
            "loop": None,
        }


def _read_animated_metadata(cache: Path) -> dict | None:
    meta_path = cache / "animated_meta.json"
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def _write_animated_metadata(cache: Path, metadata: dict) -> None:
    meta_path = cache / "animated_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        json.dump(metadata, f)


def _load_img_or_first_frame_in_rgb(path: Path) -> Image.Image:
    """
    Opens 'path' and returns a PIL Image in RGB mode, memory safe
    """
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)

        if getattr(img, "is_animated", False):
            img.seek(0)

        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            base = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(base, img.convert("RGBA")).convert("RGB")
        else:
            img = img.convert("RGB")

    return img.copy()


def check_wall(wall: Path, filter_size: tuple[int, int], threshold: float) -> bool:
    img = _load_img_or_first_frame_in_rgb(wall)
    width, height = img.size
    return width >= filter_size[0] * threshold and height >= filter_size[1] * threshold


def get_wallpaper() -> str:
    try:
        return wallpaper_path_path.read_text()
    except IOError:
        return None


def get_wallpapers(args: Namespace) -> list[Path]:
    dir = Path(args.random)
    if not dir.is_dir():
        return []

    walls = [f for f in dir.rglob("*") if is_valid_image(f)]

    if args.no_filter:
        return walls

    monitors = message("monitors")
    filter_size = min(m["width"] for m in monitors), min(m["height"] for m in monitors)

    return [f for f in walls if check_wall(f, filter_size, args.threshold)]


def get_thumb(wall: Path, cache: Path) -> Path:
    thumb = cache / "thumbnail.jpg"

    if not thumb.exists():
        img = _load_img_or_first_frame_in_rgb(wall)
        img.thumbnail((128, 128), Image.NEAREST)
        thumb.parent.mkdir(parents=True, exist_ok=True)
        img.save(thumb, "JPEG")

    return thumb


def get_smart_opts(wall: Path, cache: Path) -> str:
    opts_cache = cache / "smart.json"

    try:
        return json.loads(opts_cache.read_text())
    except (IOError, json.JSONDecodeError):
        pass

    from caelestia.utils.colourfulness import get_variant

    opts = {}

    with Image.open(get_thumb(wall, cache)) as img:
        opts["variant"] = get_variant(img)

        img.thumbnail((1, 1), Image.LANCZOS)
        hct = Hct.from_int(argb_from_rgb(*img.getpixel((0, 0))))
        opts["mode"] = "light" if hct.tone > 60 else "dark"

    opts_cache.parent.mkdir(parents=True, exist_ok=True)
    with opts_cache.open("w") as f:
        json.dump(opts, f)

    return opts


def get_colours_for_wall(wall: Path | str, no_smart: bool) -> dict:
    scheme = get_scheme()
    cache = wallpapers_cache_dir / compute_hash(wall)

    name = "dynamic"

    if not no_smart:
        smart_opts = get_smart_opts(wall, cache)
        scheme = Scheme(
            {
                "name": name,
                "flavour": "default",
                "mode": smart_opts["mode"],
                "variant": smart_opts["variant"],
                "colours": scheme.colours,
            }
        )

    return {
        "name": name,
        "flavour": "default",
        "mode": scheme.mode,
        "variant": scheme.variant,
        "colours": get_colours_for_image(get_thumb(wall, cache), scheme),
    }


def set_wallpaper(wall: Path | str, no_smart: bool) -> None:
    # Make path absolute
    wall = Path(wall).resolve()

    if not is_valid_image(wall):
        raise ValueError(f'"{wall}" is not a valid image')

    # Update files
    wallpaper_path_path.parent.mkdir(parents=True, exist_ok=True)
    wallpaper_path_path.write_text(str(wall))
    wallpaper_link_path.parent.mkdir(parents=True, exist_ok=True)
    wallpaper_link_path.unlink(missing_ok=True)
    wallpaper_link_path.symlink_to(wall)

    cache = wallpapers_cache_dir / compute_hash(wall)

    metadata = _read_animated_metadata(cache)
    if not metadata:
        metadata = _extract_animated_metadata(wall)
        # only write metadata for animated images to preserve current behaviour
        if metadata.get("is_animated"):
            _write_animated_metadata(cache, metadata)

    # Generate thumbnail or get from cache
    thumb = get_thumb(wall, cache)
    wallpaper_thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    wallpaper_thumbnail_path.unlink(missing_ok=True)
    wallpaper_thumbnail_path.symlink_to(thumb)

    scheme = get_scheme()

    # Change mode and variant based on wallpaper colour
    if scheme.name == "dynamic" and not no_smart:
        smart_opts = get_smart_opts(wall, cache)
        scheme.mode = smart_opts["mode"]
        scheme.variant = smart_opts["variant"]

    # Update colours
    scheme.update_colours()
    apply_colours(scheme.colours, scheme.mode)

    # Run custom post-hook if configured
    try:
        cfg = json.loads(user_config_path.read_text()).get("wallpaper", {})
        if post_hook := cfg.get("postHook"):
            subprocess.run(
                post_hook,
                shell=True,
                env={**os.environ, "WALLPAPER_PATH": str(wall)},
                stderr=subprocess.DEVNULL,
            )
    except (FileNotFoundError, json.JSONDecodeError):
        pass


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

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

from caelestia.utils.hypr import message
from caelestia.utils.material import get_colours_for_image
from caelestia.utils.colourfulness import get_variant
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


def is_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in [".mp4", ".webm", ".mkv"]


def djb2_hash(s: str) -> str:
    h = 5381
    for char in s:
        h = ((h << 5) + h) + ord(char)
        h &= 0xFFFFFFFF
    return f"{h:x}"


def check_wall(wall: Path, filter_size: tuple[int, int], threshold: float) -> bool:
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

    walls = [f for f in directory.rglob("*") if is_valid_image(f)]

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

    if wall.suffix.lower() == ".gif":
        wall_cache = convert_gif(wall)
    elif is_video(wall):
        wall_cache = convert_video(wall)
    else:
        wall_cache = wall

    cache = wallpapers_cache_dir / compute_hash(wall_cache)

    name = "dynamic"

    if not no_smart:
        smart_opts = get_smart_opts(wall_cache, cache)
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
        "colours": get_colours_for_image(get_thumb(wall_cache, cache), scheme),
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

# AW implementation 
def convert_video(wall: Path) -> Path:
    from caelestia.utils.paths import c_cache_dir
    # Check for pre-generated fast thumbnail
    fast_thumb = c_cache_dir / "videothumbs" / f"{djb2_hash(str(wall.resolve()))}.jpg"
    if fast_thumb.exists():
        return fast_thumb

    cache = wallpapers_cache_dir / compute_hash(wall)
    output_path = cache / "first_frame.png"

    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-ss", "00:00:00.5",
            "-i", str(wall),
            "-vframes", "1",
            "-q:v", "2",
            str(output_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            # Fallback to 00:00:00
            cmd[3] = "00:00:00"
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except Exception:
                pass

    return output_path


def extract_all_video_thumbs() -> None:
    from caelestia.utils.paths import wallpapers_dir, c_cache_dir
    from concurrent.futures import ThreadPoolExecutor
    import threading

    videothumbs_dir = c_cache_dir / "videothumbs"
    videothumbs_dir.mkdir(parents=True, exist_ok=True)
    
    ready_file = Path("/tmp/caelestia_thumb_ready.txt")
    if ready_file.exists():
        try:
            ready_file.unlink()
        except OSError:
            pass

    write_lock = threading.Lock()

    def process_video(file_path: Path):
        try:
            resolved_path = file_path.resolve()
            h = djb2_hash(str(resolved_path))
            thumb_path = videothumbs_dir / f"{h}.jpg"
            
            
            if not thumb_path.exists():
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", "00:00:00.5",
                    "-i", str(resolved_path),
                    "-vframes", "1",
                    "-q:v", "4",
                    str(thumb_path)
                ]
                
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    success = True
                except Exception:
                    cmd[3] = "00:00:00"
                    try:
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        success = True
                    except Exception:
                        success = False
            else:
                success = True
                
            if success:
                with write_lock:
                    with open(ready_file, "a") as f:
                        f.write(f"file://{file_path}\n")
                        
        except Exception:
            pass

    video_extensions = {".mp4", ".webm", ".mkv"}
    videos_to_process = []
    
    for root_dir, _, files in os.walk(wallpapers_dir):
        for file in files:
            file_path = Path(root_dir) / file
            if file_path.suffix.lower() in video_extensions:
                videos_to_process.append(file_path)
                
    with ThreadPoolExecutor(max_workers=16) as executor:
        for _ in executor.map(process_video, videos_to_process):
            pass


def set_wallpaper(wall: Path, no_smart: bool) -> None:
    # Make path absolute
    wall = Path(wall).resolve()

    if not is_valid_image(wall) and not is_video(wall):
        raise ValueError(f'"{wall}" is not a valid image or video')

    # Use gif's 1st frame or video's extracted frame for thumb only
    if wall.suffix.lower() == ".gif":
        wall_cache = convert_gif(wall)
    elif is_video(wall):
        wall_cache = convert_video(wall)
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
    print(f"DEBUG: setting wallpaper to {wall}")
    wallpaper_thumbnail_path.symlink_to(thumb)

    if is_video(wall):
        from caelestia.utils.paths import c_cache_dir
        videothumbs_dir = c_cache_dir / "videothumbs"
        videothumbs_dir.mkdir(parents=True, exist_ok=True)
        fast_thumb = videothumbs_dir / f"{djb2_hash(str(wall.resolve()))}.jpg"
        if not fast_thumb.exists():
            import shutil
            try:
                shutil.copy(thumb, fast_thumb)
            except Exception:
                pass

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
    try:
        cfg = json.loads(user_config_path.read_text()).get("wallpaper", {})
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



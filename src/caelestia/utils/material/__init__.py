import json
from pathlib import Path

from materialyoucolor.hct import Hct

from caelestia.utils.material.generator import gen_scheme
from caelestia.utils.material.score import score
from caelestia.utils.paths import compute_hash, scheme_cache_dir, wallpaper_thumbnail_path


def get_score_for_image(image: str, cache_base: Path) -> tuple[list[Hct], list[Hct]]:
    cache = cache_base / "score.json"

    try:
        with cache.open("r") as f:
            return [[Hct.from_int(c) for c in li] for li in json.load(f)]
    except (IOError, json.JSONDecodeError):
        pass

    s = score(image)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        json.dump([[c.to_int() for c in li] for li in s], f)

    return s


def get_colours_for_image(image: str = str(wallpaper_thumbnail_path), scheme=None) -> dict[str, str]:
    if scheme is None:
        from caelestia.utils.scheme import get_scheme

        scheme = get_scheme()

    cache_base = scheme_cache_dir / compute_hash(image)
    cache = (cache_base / scheme.variant / scheme.flavour / scheme.mode).with_suffix(".json")

    try:
        with cache.open("r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        pass

    primaries, colours = get_score_for_image(image, cache_base)
    i = ["default", "alt1", "alt2"].index(scheme.flavour)
    scheme = gen_scheme(scheme, primaries[i], colours)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        json.dump(scheme, f)

    return scheme

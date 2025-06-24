import json
from pathlib import Path

from caelestia.utils.paths import compute_hash, scheme_cache_dir, wallpaper_thumbnail_path


def get_score_for_image(image: Path | str, cache_base: Path):
    from materialyoucolor.hct import Hct

    cache = cache_base / "score.json"

    try:
        return Hct.from_int(cache.read_text())
    except (IOError, TypeError):
        pass

    from caelestia.utils.material.score import score

    s = score(str(image))

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(str(s.to_int()))

    return s


def get_colours_for_image(image: Path | str = wallpaper_thumbnail_path, scheme=None) -> dict[str, str]:
    if scheme is None:
        from caelestia.utils.scheme import get_scheme

        scheme = get_scheme()

    cache_base = scheme_cache_dir / compute_hash(image)
    cache = (cache_base / scheme.variant / scheme.mode).with_suffix(".json")

    try:
        with cache.open("r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        pass

    from caelestia.utils.material.generator import gen_scheme

    primary = get_score_for_image(image, cache_base)
    scheme = gen_scheme(scheme, primary)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        json.dump(scheme, f)

    return scheme

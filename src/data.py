import os
from pathlib import Path

config_dir = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
data_dir = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local/share"))
state_dir = Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local/state"))

c_config_dir = config_dir / "caelestia"
c_data_dir = data_dir / "caelestia"
c_state_dir = state_dir / "caelestia"

scheme_name_path = c_state_dir / "scheme/name.txt"
scheme_flavour_path = c_state_dir / "scheme/flavour.txt"
scheme_colours_path = c_state_dir / "scheme/colours.txt"
scheme_mode_path = c_state_dir / "scheme/mode.txt"
scheme_variant_path = c_state_dir / "scheme/variant.txt"

scheme_data_path = Path(__file__).parent.parent / "data/schemes"

scheme_variants = [
    "tonalspot",
    "vibrant",
    "expressive",
    "fidelity",
    "fruitsalad",
    "monochrome",
    "neutral",
    "rainbow",
    "content",
]

scheme_names: list[str] = None
scheme_flavours: list[str] = None
scheme_modes: list[str] = None

scheme_name: str = None
scheme_flavour: str = None
scheme_colours: dict[str, str] = None
scheme_mode: str = None
scheme_variant: str = None


def get_scheme_path() -> Path:
    return (scheme_data_path / get_scheme_name() / get_scheme_flavour() / get_scheme_mode()).with_suffix(".txt")


def get_scheme_names() -> list[str]:
    global scheme_names

    if scheme_names is None:
        scheme_names = [f.name for f in scheme_data_path.iterdir() if f.is_dir()]

    return scheme_names


def get_scheme_flavours() -> list[str]:
    global scheme_flavours

    if scheme_flavours is None:
        scheme_flavours = [f.name for f in (scheme_data_path / get_scheme_name()).iterdir() if f.is_dir()]

    return scheme_flavours


def get_scheme_modes() -> list[str]:
    global scheme_modes

    if scheme_modes is None:
        scheme_modes = [
            f.stem for f in (scheme_data_path / get_scheme_name() / get_scheme_flavour()).iterdir() if f.is_file()
        ]

    return scheme_modes


def get_scheme_name() -> str:
    global scheme_name

    if scheme_name is None:
        scheme_name = scheme_name_path.read_text().strip() if scheme_name_path.exists() else "catppuccin"

    return scheme_name


def get_scheme_flavour() -> str:
    global scheme_flavour

    if scheme_flavour is None:
        scheme_flavour = scheme_flavour_path.read_text().strip() if scheme_flavour_path.exists() else "mocha"

    return scheme_flavour


def get_scheme_colours() -> dict[str, str]:
    global scheme_colours

    if scheme_colours is None:
        scheme_colours = {
            k.strip(): v.strip() for k, v in (line.split(" ") for line in get_scheme_path().read_text().splitlines())
        }

    return scheme_colours


def get_scheme_mode() -> str:
    global scheme_mode

    if scheme_mode is None:
        scheme_mode = scheme_mode_path.read_text().strip() if scheme_mode_path.exists() else "dark"

    return scheme_mode


def get_scheme_variant() -> str:
    global scheme_variant

    if scheme_variant is None:
        scheme_variant = scheme_variant_path.read_text().strip() if scheme_variant_path.exists() else "tonalspot"

    return scheme_variant

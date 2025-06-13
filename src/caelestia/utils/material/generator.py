from materialyoucolor.blend import Blend
from materialyoucolor.dynamiccolor.material_dynamic_colors import (
    DynamicScheme,
    MaterialDynamicColors,
)
from materialyoucolor.hct import Hct
from materialyoucolor.hct.cam16 import Cam16
from materialyoucolor.scheme.scheme_content import SchemeContent
from materialyoucolor.scheme.scheme_expressive import SchemeExpressive
from materialyoucolor.scheme.scheme_fidelity import SchemeFidelity
from materialyoucolor.scheme.scheme_fruit_salad import SchemeFruitSalad
from materialyoucolor.scheme.scheme_monochrome import SchemeMonochrome
from materialyoucolor.scheme.scheme_neutral import SchemeNeutral
from materialyoucolor.scheme.scheme_rainbow import SchemeRainbow
from materialyoucolor.scheme.scheme_tonal_spot import SchemeTonalSpot
from materialyoucolor.scheme.scheme_vibrant import SchemeVibrant


def hex_to_hct(hex: str) -> Hct:
    return Hct.from_int(int(f"0xFF{hex}", 16))


light_colours = [
    hex_to_hct("dc8a78"),
    hex_to_hct("dd7878"),
    hex_to_hct("ea76cb"),
    hex_to_hct("8839ef"),
    hex_to_hct("d20f39"),
    hex_to_hct("e64553"),
    hex_to_hct("fe640b"),
    hex_to_hct("df8e1d"),
    hex_to_hct("40a02b"),
    hex_to_hct("179299"),
    hex_to_hct("04a5e5"),
    hex_to_hct("209fb5"),
    hex_to_hct("1e66f5"),
    hex_to_hct("7287fd"),
]

dark_colours = [
    hex_to_hct("f5e0dc"),
    hex_to_hct("f2cdcd"),
    hex_to_hct("f5c2e7"),
    hex_to_hct("cba6f7"),
    hex_to_hct("f38ba8"),
    hex_to_hct("eba0ac"),
    hex_to_hct("fab387"),
    hex_to_hct("f9e2af"),
    hex_to_hct("a6e3a1"),
    hex_to_hct("94e2d5"),
    hex_to_hct("89dceb"),
    hex_to_hct("74c7ec"),
    hex_to_hct("89b4fa"),
    hex_to_hct("b4befe"),
]

colour_names = [
    "rosewater",
    "flamingo",
    "pink",
    "mauve",
    "red",
    "maroon",
    "peach",
    "yellow",
    "green",
    "teal",
    "sky",
    "sapphire",
    "blue",
    "lavender",
    "success",
    "error",
]


def grayscale(colour: Hct, light: bool) -> Hct:
    colour = darken(colour, 0.35) if light else lighten(colour, 0.65)
    colour.chroma = 0
    return colour


def mix(a: Hct, b: Hct, w: float) -> Hct:
    return Hct.from_int(Blend.cam16_ucs(a.to_int(), b.to_int(), w))


def harmonize(a: Hct, b: Hct) -> Hct:
    return Hct.from_int(Blend.harmonize(a.to_int(), b.to_int()))


def lighten(colour: Hct, amount: float) -> Hct:
    diff = (100 - colour.tone) * amount
    return Hct.from_hct(colour.hue, colour.chroma + diff / 5, colour.tone + diff)


def darken(colour: Hct, amount: float) -> Hct:
    diff = colour.tone * amount
    return Hct.from_hct(colour.hue, colour.chroma + diff / 5, colour.tone - diff)


def distance(colour: Cam16, base: Cam16) -> float:
    return colour.distance(base)


def smart_sort(colours: list[Hct], base: list[Hct]) -> dict[str, Hct]:
    sorted_colours = [None] * len(colours)
    distances = {}

    cams = [(c, Cam16.from_int(c.to_int())) for c in colours]
    base_cams = [Cam16.from_int(c.to_int()) for c in base]

    for colour, cam in cams:
        dist = [(i, distance(cam, b)) for i, b in enumerate(base_cams)]
        dist.sort(key=lambda x: x[1])
        distances[colour] = dist

    for colour in colours:
        while len(distances[colour]) > 0:
            i, dist = distances[colour][0]

            if sorted_colours[i] is None:
                sorted_colours[i] = colour, dist
                break
            elif sorted_colours[i][1] > dist:
                old = sorted_colours[i][0]
                sorted_colours[i] = colour, dist
                colour = old

            distances[colour].pop(0)

    return {colour_names[i]: c[0] for i, c in enumerate(sorted_colours)}


def get_scheme(scheme: str) -> DynamicScheme:
    if scheme == "content":
        return SchemeContent
    if scheme == "expressive":
        return SchemeExpressive
    if scheme == "fidelity":
        return SchemeFidelity
    if scheme == "fruitsalad":
        return SchemeFruitSalad
    if scheme == "monochrome":
        return SchemeMonochrome
    if scheme == "neutral":
        return SchemeNeutral
    if scheme == "rainbow":
        return SchemeRainbow
    if scheme == "tonalspot":
        return SchemeTonalSpot
    return SchemeVibrant


def gen_scheme(scheme, primary: Hct, colours: list[Hct]) -> dict[str, str]:
    light = scheme.mode == "light"
    base = light_colours if light else dark_colours

    # Sort colours and turn into dict
    colours = smart_sort(colours, base)

    # Harmonize colours
    for name, hct in colours.items():
        if scheme.variant == "monochrome":
            colours[name] = grayscale(hct, light)
        else:
            harmonized = harmonize(hct, primary)
            colours[name] = darken(harmonized, 0.35) if light else lighten(harmonized, 0.65)

    # Material colours
    primary_scheme = get_scheme(scheme.variant)(primary, not light, 0)
    for colour in vars(MaterialDynamicColors).keys():
        colour_name = getattr(MaterialDynamicColors, colour)
        if hasattr(colour_name, "get_hct"):
            colours[colour] = colour_name.get_hct(primary_scheme)

    # FIXME: deprecated stuff
    colours["text"] = colours["onBackground"]
    colours["subtext1"] = colours["onSurfaceVariant"]
    colours["subtext0"] = colours["outline"]
    colours["overlay2"] = mix(colours["surface"], colours["outline"], 0.86)
    colours["overlay1"] = mix(colours["surface"], colours["outline"], 0.71)
    colours["overlay0"] = mix(colours["surface"], colours["outline"], 0.57)
    colours["surface2"] = mix(colours["surface"], colours["outline"], 0.43)
    colours["surface1"] = mix(colours["surface"], colours["outline"], 0.29)
    colours["surface0"] = mix(colours["surface"], colours["outline"], 0.14)
    colours["base"] = colours["surface"]
    colours["mantle"] = darken(colours["surface"], 0.03)
    colours["crust"] = darken(colours["surface"], 0.05)
    colours["success"] = harmonize(base[8], primary)

    # For debugging
    # print("\n".join(["{}: \x1b[48;2;{};{};{}m   \x1b[0m".format(n, *c.to_rgba()[:3]) for n, c in colours.items()]))

    return {k: hex(v.to_int())[4:] for k, v in colours.items()}

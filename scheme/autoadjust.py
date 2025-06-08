#!/usr/bin/env python3

import sys
from colorsys import hls_to_rgb, rgb_to_hls
from pathlib import Path

from materialyoucolor.blend import Blend
from materialyoucolor.dynamiccolor.material_dynamic_colors import (
    DynamicScheme,
    MaterialDynamicColors,
)
from materialyoucolor.hct import Hct
from materialyoucolor.scheme.scheme_content import SchemeContent
from materialyoucolor.scheme.scheme_expressive import SchemeExpressive
from materialyoucolor.scheme.scheme_fidelity import SchemeFidelity
from materialyoucolor.scheme.scheme_fruit_salad import SchemeFruitSalad
from materialyoucolor.scheme.scheme_monochrome import SchemeMonochrome
from materialyoucolor.scheme.scheme_neutral import SchemeNeutral
from materialyoucolor.scheme.scheme_rainbow import SchemeRainbow
from materialyoucolor.scheme.scheme_tonal_spot import SchemeTonalSpot
from materialyoucolor.scheme.scheme_vibrant import SchemeVibrant

light_colours = [
    "dc8a78",
    "dd7878",
    "ea76cb",
    "8839ef",
    "d20f39",
    "e64553",
    "fe640b",
    "df8e1d",
    "40a02b",
    "179299",
    "04a5e5",
    "209fb5",
    "1e66f5",
    "7287fd",
]

dark_colours = [
    "f5e0dc",
    "f2cdcd",
    "f5c2e7",
    "cba6f7",
    "f38ba8",
    "eba0ac",
    "fab387",
    "f9e2af",
    "a6e3a1",
    "94e2d5",
    "89dceb",
    "74c7ec",
    "89b4fa",
    "b4befe",
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

HLS = tuple[float, float, float]


def hex_to_rgb(hex: str) -> tuple[float, float, float]:
    """Convert a hex string to an RGB tuple in the range [0, 1]."""
    return tuple(int(hex[i : i + 2], 16) / 255 for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    """Convert an RGB tuple in the range [0, 1] to a hex string."""
    return "".join(f"{round(i * 255):02X}" for i in rgb)


def hex_to_hls(hex: str) -> tuple[float, float, float]:
    return rgb_to_hls(*hex_to_rgb(hex))


def hls_to_hex(h: str, l: str, s: str) -> str:
    return rgb_to_hex(hls_to_rgb(h, l, s))


def hex_to_argb(hex: str) -> int:
    return int(f"0xFF{hex}", 16)


def argb_to_hls(argb: int) -> HLS:
    return hex_to_hls(f"{argb:08X}"[2:])


def grayscale(hls: HLS, light: bool) -> HLS:
    h, l, s = hls
    return h, 0.5 - l / 2 if light else l / 2 + 0.5, 0


def mix(a: HLS, b: HLS, w: float) -> HLS:
    r1, g1, b1 = hls_to_rgb(*a)
    r2, g2, b2 = hls_to_rgb(*b)
    return rgb_to_hls(
        r1 * (1 - w) + r2 * w, g1 * (1 - w) + g2 * w, b1 * (1 - w) + b2 * w
    )


def harmonize(a: str, b: int) -> HLS:
    return argb_to_hls(Blend.harmonize(hex_to_argb(a), b))


def darken(colour: HLS, amount: float) -> HLS:
    h, l, s = colour
    return h, max(0, l - amount), s


def distance(colour: HLS, base: str) -> float:
    h1, l1, s1 = colour
    h2, l2, s2 = hex_to_hls(base)
    return abs(h1 - h2) * 0.4 + abs(l1 - l2) * 0.3 + abs(s1 - s2) * 0.3


def smart_sort(colours: list[HLS], base: list[str]) -> dict[str, HLS]:
    sorted_colours = [None] * len(colours)
    distances = {}

    for colour in colours:
        dist = [(i, distance(colour, b)) for i, b in enumerate(base)]
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


def get_alt(i: int) -> str:
    names = ["default", "alt1", "alt2"]
    return names[i]


if __name__ == "__main__":
    light = sys.argv[1] == "light"
    scheme = sys.argv[2]
    primaries = sys.argv[3].split(" ")
    colours_in = sys.argv[4].split(" ")
    out_path = sys.argv[5]

    base = light_colours if light else dark_colours

    # Convert to HLS
    base_colours = [hex_to_hls(c) for c in colours_in]

    # Sort colours and turn into dict
    base_colours = smart_sort(base_colours, base)

    # Adjust colours
    MatScheme = get_scheme(scheme)
    for name, hls in base_colours.items():
        if scheme == "monochrome":
            base_colours[name] = grayscale(hls, light)
        else:
            argb = hex_to_argb(hls_to_hex(*hls))
            mat_scheme = MatScheme(Hct.from_int(argb), not light, 0)

            colour = MaterialDynamicColors.primary.get_hct(mat_scheme)

            # Boost neutral scheme colours
            if scheme == "neutral":
                colour.chroma += 10

            base_colours[name] = hex_to_hls(
                "{:02X}{:02X}{:02X}".format(*colour.to_rgba()[:3])
            )

    # Layers and accents
    for i, primary in enumerate(primaries):
        material = {}

        primary_argb = hex_to_argb(primary)
        primary_scheme = MatScheme(Hct.from_int(primary_argb), not light, 0)
        for colour in vars(MaterialDynamicColors).keys():
            colour_name = getattr(MaterialDynamicColors, colour)
            if hasattr(colour_name, "get_hct"):
                rgb = colour_name.get_hct(primary_scheme).to_rgba()[:3]
                material[colour] = hex_to_hls("{:02X}{:02X}{:02X}".format(*rgb))

        # TODO: eventually migrate to material for layers
        colours = {
            **material,
            "text": material["onBackground"],
            "subtext1": material["onSurfaceVariant"],
            "subtext0": material["outline"],
            "overlay2": mix(material["surface"], material["outline"], 0.86),
            "overlay1": mix(material["surface"], material["outline"], 0.71),
            "overlay0": mix(material["surface"], material["outline"], 0.57),
            "surface2": mix(material["surface"], material["outline"], 0.43),
            "surface1": mix(material["surface"], material["outline"], 0.29),
            "surface0": mix(material["surface"], material["outline"], 0.14),
            "base": material["surface"],
            "mantle": darken(material["surface"], 0.03),
            "crust": darken(material["surface"], 0.05),
            "success": harmonize(base[8], primary_argb),
        }

        for name, hls in base_colours.items():
            colours[name] = harmonize(hls_to_hex(*hls), primary_argb)

        out_file = Path(f"{out_path}/{scheme}/{get_alt(i)}/{sys.argv[1]}.txt")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        colour_arr = [
            f"{name} {hls_to_hex(*colour)}" for name, colour in colours.items()
        ]
        out_file.write_text("\n".join(colour_arr))

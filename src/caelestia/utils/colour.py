class Colour:
    _rgb_vals: tuple[int, ...]
    _hex_vals: tuple[str, ...]
    _hsl_vals: tuple[int, int, int]

    def __init__(self, hex: str):
        hex = hex.ljust(8, "f")
        self._hex_vals = tuple(hex[i : i + 2] for i in range(0, 7, 2))
        self._rgb_vals = tuple(int(h, 16) for h in self._hex_vals)
        self._hsl_vals = self._to_hsl(self._rgb_vals)

    @property
    def hex(self) -> str:
        return "".join(self._hex_vals[:-1])

    @property
    def hexalpha(self) -> str:
        return "".join(self._hex_vals)

    @property
    def rgb(self) -> str:
        return f"rgb({','.join(map(str, self._rgb_vals[:-1]))})"

    @property
    def rgbalpha(self) -> str:
        return f"rgba({','.join(map(str, self._rgb_vals))})"

    @property
    def red(self) -> int:
        return self._rgb_vals[0]

    @property
    def green(self) -> int:
        return self._rgb_vals[1]

    @property
    def blue(self) -> int:
        return self._rgb_vals[2]

    @property
    def hsl(self) -> str:
        return f"hsl({self._hsl_vals[0]},{self._hsl_vals[1]}%,{self._hsl_vals[2]}%)"

    @property
    def hue(self) -> int:
        return self._hsl_vals[0]

    @property
    def saturation(self) -> int:
        return self._hsl_vals[1]

    @property
    def lightness(self) -> int:
        return self._hsl_vals[2]

    @staticmethod
    def _to_hsl(rgb: tuple[int, ...]) -> tuple[int, int, int]:
        r, g, b = (v / 255 for v in rgb[:3])
        cmax, cmin = max(r, g, b), min(r, g, b)
        delta = cmax - cmin

        lightness = (cmax + cmin) / 2
        saturation = 0 if delta == 0 else delta / (1 - abs(2 * lightness - 1))

        if delta == 0:
            hue = 0
        elif cmax == r:
            hue = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            hue = 60 * (((b - r) / delta) + 2)
        else:
            hue = 60 * (((r - g) / delta) + 4)

        return round(hue), round(saturation * 100), round(lightness * 100)


def get_dynamic_colours(colours: dict[str, str]) -> dict[str, Colour]:
    return {name: Colour(code) for name, code in colours.items()}

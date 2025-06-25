import math

from PIL import Image


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0


def stddev(values: list[float], mean_val: float) -> float:
    return math.sqrt(sum((x - mean_val) ** 2 for x in values) / len(values)) if values else 0


def calc_colourfulness(image: Image) -> float:
    width, height = image.size
    pixels = list(image.getdata())  # List of (R, G, B) tuples

    rg_diffs = []
    yb_diffs = []

    for r, g, b in pixels:
        rg = abs(r - g)
        yb = abs(0.5 * (r + g) - b)
        rg_diffs.append(rg)
        yb_diffs.append(yb)

    mean_rg = mean(rg_diffs)
    mean_yb = mean(yb_diffs)
    std_rg = stddev(rg_diffs, mean_rg)
    std_yb = stddev(yb_diffs, mean_yb)

    return math.sqrt(std_rg**2 + std_yb**2) + 0.3 * math.sqrt(mean_rg**2 + mean_yb**2)


def get_variant(image: Image) -> str:
    colourfulness = calc_colourfulness(image)

    if colourfulness < 10:
        return "neutral"
    if colourfulness < 20:
        return "content"
    return "tonalspot"

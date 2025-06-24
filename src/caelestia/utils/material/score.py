from materialyoucolor.dislike.dislike_analyzer import DislikeAnalyzer
from materialyoucolor.hct import Hct
from materialyoucolor.quantize import ImageQuantizeCelebi
from materialyoucolor.utils.math_utils import sanitize_degrees_int


class Score:
    TARGET_CHROMA = 48.0
    WEIGHT_PROPORTION = 0.7
    WEIGHT_CHROMA_ABOVE = 0.3
    WEIGHT_CHROMA_BELOW = 0.1
    CUTOFF_CHROMA = 5.0
    CUTOFF_EXCITED_PROPORTION = 0.01

    def __init__(self):
        pass

    @staticmethod
    def score(colors_to_population: dict, filter_enabled: bool = False) -> Hct:
        colors_hct = []
        hue_population = [0] * 360
        population_sum = 0

        for rgb, population in colors_to_population.items():
            hct = Hct.from_int(rgb)
            colors_hct.append(hct)
            hue = int(hct.hue)
            hue_population[hue] += population
            population_sum += population

        hue_excited_proportions = [0.0] * 360

        for hue in range(360):
            proportion = hue_population[hue] / population_sum
            for i in range(hue - 14, hue + 16):
                neighbor_hue = int(sanitize_degrees_int(i))
                hue_excited_proportions[neighbor_hue] += proportion

        # Score colours
        scored_hct = []
        for hct in colors_hct:
            hue = int(sanitize_degrees_int(round(hct.hue)))
            proportion = hue_excited_proportions[hue]

            if filter_enabled and (hct.chroma < Score.CUTOFF_CHROMA or proportion <= Score.CUTOFF_EXCITED_PROPORTION):
                continue

            proportion_score = proportion * 100.0 * Score.WEIGHT_PROPORTION
            chroma_weight = Score.WEIGHT_CHROMA_BELOW if hct.chroma < Score.TARGET_CHROMA else Score.WEIGHT_CHROMA_ABOVE
            chroma_score = (hct.chroma - Score.TARGET_CHROMA) * chroma_weight
            score = proportion_score + chroma_score
            scored_hct.append({"hct": hct, "score": score})

        scored_hct.sort(key=lambda x: x["score"], reverse=True)

        # Get primary colour
        primary = None
        for cutoff in range(20, -1, -1):
            for item in scored_hct:
                if item["hct"].chroma > cutoff and item["hct"].tone > cutoff * 3:
                    primary = item["hct"]
                    break
            if primary:
                break

        return DislikeAnalyzer.fix_if_disliked(primary) if primary else Score.score(colors_to_population, False)


def score(image: str) -> Hct:
    return Score.score(ImageQuantizeCelebi(image, 1, 128))

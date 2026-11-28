import json
import subprocess
from argparse import Namespace
from urllib.request import urlopen

from caelestia.utils.deprecation import deprecated_in_favour_of_global
from caelestia.utils.paths import cli_data_dir


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.picker:
            deprecated_in_favour_of_global("emoji -p", "emoji")

            emojis = (cli_data_dir / "emojis.txt").read_text()
            chosen = subprocess.check_output(
                ["fuzzel", "--dmenu", "--placeholder=Type to search emojis"], input=emojis, text=True
            )
            subprocess.run(["wl-copy"], input=chosen.split()[0], text=True)
        elif self.args.fetch:
            self.fetch_emojis()
        else:
            print((cli_data_dir / "emojis.txt").read_text(), end="")

    def fetch_emojis(self) -> None:
        data = [
            "¿? question upside down reversed spanish",
            "← left arrow",
            "↑ up arrow",
            "→ right arrow",
            "↓ down arrow",
            "←↑→↓ all directions up down left right arrows",
            "⇇ leftwards paired arrows",
            "⇉ rightwards paired arrows",
            "⇈ upwards paired arrows",
            "⇊ downwards paired arrows",
            "⬱ three leftwards arrows",
            "⇶ three rightwards arrows",
            "• dot circle separator",
            "「」 japanese quote square bracket",
            "¯\\_(ツ)_/¯ shrug idk i dont know",
            "(ง🔥ﾛ🔥)ง person with fire eyes eyes on fire",
            "↵ enter key return",
            "° degrees",
            "™ tm trademark",
            "® registered trademark",
            "© copyright",
            "— em dash",
            "󰖳 windows super key",
        ]

        # Fetch emojis
        with urlopen(
            "https://raw.githubusercontent.com/milesj/emojibase/refs/heads/master/packages/data/en/compact.raw.json"
        ) as f:
            emojis = json.load(f)

        for emoji in emojis:
            line = [emoji["unicode"]]

            if "emoticon" in emoji:
                if isinstance(emoji["emoticon"], str):
                    line.append(emoji["emoticon"])
                else:
                    line.extend(emoji["emoticon"])

            line.append(emoji["label"])

            if "tags" in emoji:
                line.extend(emoji["tags"])

            data.append(" ".join(line))

        # Fetch nerd font glyphs
        with urlopen("https://raw.githubusercontent.com/ryanoasis/nerd-fonts/refs/heads/master/glyphnames.json") as f:
            glyphs = json.load(f)

        buckets = {}
        for name, glyph in glyphs.items():
            if name == "METADATA":
                continue

            unicode = glyph["char"]
            if unicode not in buckets:
                buckets[unicode] = []
            buckets[unicode].append(f"nf-{name}")

        for glyph, names in buckets.items():
            data.append(f"{glyph}  {' '.join(names)}")

        # Write to file
        (cli_data_dir / "emojis.txt").write_text("\n".join(data))

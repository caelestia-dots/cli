import os
from argparse import Namespace

from caelestia.utils.paths import neovim_colors_path, neovim_colorscheme_file, neovim_template_path
from caelestia.utils.scheme import get_scheme


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        # 1. Get the scheme
        scheme = get_scheme()

        # 2. Read the template
        template_file = neovim_template_path
        with open(template_file, "r") as f:
            template = f.read()

        # 3. Replace placeholders
        for color_name, color_value in scheme.colours.items():
            template = template.replace(f"{{{{ ${color_name} }}}}", color_value)

        # 4. Write the colorscheme
        os.makedirs(neovim_colors_path, exist_ok=True)

        with open(neovim_colorscheme_file, "w") as f:
            f.write(template)

        print(f"Neovim colorscheme generated at {neovim_colorscheme_file}")

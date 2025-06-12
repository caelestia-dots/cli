from argparse import Namespace

from caelestia.utils.scheme import get_scheme
from caelestia.utils.theme import apply_colours


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        scheme = get_scheme()

        if self.args.random:
            scheme.set_random()
            apply_colours(scheme.colours, scheme.mode)
        elif self.args.name or self.args.flavour or self.args.mode:
            if self.args.name:
                scheme.name = self.args.name
            if self.args.flavour:
                scheme.flavour = self.args.flavour
            if self.args.mode:
                scheme.mode = self.args.mode
            apply_colours(scheme.colours, scheme.mode)
        else:
            print(scheme)

import json
from argparse import Namespace

from caelestia.utils.wallpaper import get_colours_for_wall, get_wallpaper, set_random, set_wallpaper


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.print:
            # -p with no path means the current wallpaper (resolved here instead of
            # at argument parsing time to avoid the cost on unrelated commands)
            wall = get_wallpaper() if self.args.print is True else self.args.print
            if wall is None:
                print("No wallpaper set")
                return
            print(json.dumps(get_colours_for_wall(wall, self.args.no_smart)))
        elif self.args.file:
            set_wallpaper(self.args.file, self.args.no_smart)
        elif self.args.random:
            set_random(self.args)
        else:
            print(get_wallpaper() or "No wallpaper set")

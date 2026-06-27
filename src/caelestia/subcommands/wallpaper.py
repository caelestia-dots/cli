import json
from argparse import Namespace

from caelestia.utils.wallpaper import get_colours_for_wall, get_wallpaper, set_random, set_wallpaper, extract_all_video_thumbs


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        if self.args.print:
            print(json.dumps(get_colours_for_wall(self.args.print, self.args.no_smart)))
        elif self.args.file:
            set_wallpaper(self.args.file, self.args.no_smart)
        elif self.args.random:
            set_random(self.args)
        elif self.args.extract_thumbs:
            extract_all_video_thumbs()
        else:
            print(get_wallpaper() or "No wallpaper set")

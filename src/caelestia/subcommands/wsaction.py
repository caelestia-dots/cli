from argparse import Namespace

from caelestia.utils import hypr


class Command:
    args: Namespace

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def run(self) -> None:
        active_ws = hypr.message("activeworkspace")["id"]

        if self.args.group:
            hypr.dispatch(self.args.dispatcher, (self.args.workspace - 1) * 10 + active_ws % 10)
        else:
            hypr.dispatch(self.args.dispatcher, int((active_ws - 1) / 10) * 10 + self.args.workspace)

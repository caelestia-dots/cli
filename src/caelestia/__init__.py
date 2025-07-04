from caelestia.parser import parse_args
from caelestia.utils.version import print_version


def main() -> None:
    parser, args = parse_args()
    if args.version:
        print_version()
    elif "cls" in args:
        args.cls(args).run()
    else:
        parser.print_help()

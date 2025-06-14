from caelestia.parser import parse_args


def main() -> None:
    parser, args = parse_args()
    if "cls" in args:
        args.cls(args).run()
    else:
        parser.print_help()

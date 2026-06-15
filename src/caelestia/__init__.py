from importlib import import_module

from caelestia.parser import parse_args


def main() -> None:
    parser, args = parse_args()
    if args.version:
        from caelestia.utils.version import print_version

        print_version()
    elif "cls" in args:
        # Resolve the lazy "module:class" reference from the parser
        module, _, cls = args.cls.partition(":")
        command = getattr(import_module(f"caelestia.subcommands.{module}"), cls)
        command(args).run()
    else:
        parser.print_help()

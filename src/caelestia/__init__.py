from caelestia.parser import parse_args


def main() -> None:
    args = parse_args()
    if "cls" in args:
        args.cls(args).run()
    else:
        import sys
        print("No arguments given", file=sys.stderr)

if __name__ == "__main__":
    main()

from caelestia.parser import parse_args


def main() -> None:
    args = parse_args()
    args.cls(args).run()

if __name__ == "__main__":
    main()

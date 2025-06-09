from parser import parse_args

if __name__ == "__main__":
    args = parse_args()
    args.cls(args).run()

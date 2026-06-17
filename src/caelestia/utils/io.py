import sys
from typing import Never

LOG_COLOUR: int = 2
INFO_COLOUR: int = 0
PROMPT_COLOUR: int = 36
WARNING_COLOUR: int = 33
ERROR_COLOUR: int = 31

_disable_input: bool = False


def disable_input() -> None:
    global _disable_input
    _disable_input = True


def log_exception(func):
    """Log exceptions to stderr instead of raising.

    Used by the `apply_()` functions so that an exception, when applying
    a theme, does not prevent the other themes from being applied.
    """

    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            error(f'exception during "{func.__name__}()": {str(e)}')

    return wrapper


def format_msg(colour: int, prefix: bool, msg: str) -> str:
    return f"\033[{colour}m{':: ' if prefix else ''}{msg}\033[0m"


def log(msg: str, prefix: bool = True) -> None:
    print(format_msg(LOG_COLOUR, prefix, msg))


def info(msg: str, prefix: bool = True) -> None:
    print(format_msg(INFO_COLOUR, prefix, msg))


def warn(msg: str, prefix: bool = True) -> None:
    print(format_msg(WARNING_COLOUR, prefix, f"Warning: {msg}"))


def error(err: str | Exception, prefix: bool = True) -> None:
    print(format_msg(ERROR_COLOUR, prefix, f"Error: {err}"), file=sys.stderr)


def fatal(err: str | Exception, prefix: bool = True) -> Never:
    print(format_msg(ERROR_COLOUR, prefix, f"Fatal: {err}"), file=sys.stderr)
    sys.exit(1)


def _input(prompt: str) -> str:
    if _disable_input:
        print(prompt, end="")
        return ""

    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        print()
        raise KeyboardInterrupt()


def prompt(msg: str, prefix: bool = True, end: str = " ") -> str:
    return _input(format_msg(PROMPT_COLOUR, prefix, msg) + end)


def confirm(msg: str, prefix: bool = True, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    answer = prompt(msg + suffix, prefix=prefix).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def prompt_selection(items: list[str], header: str) -> list[str]:
    """Prompt the user to pick from a numbered list, returning the selected items.

    Accepts `[A]ll`/`a`, single indices, ranges (`1-3`) and exclusions (`^4`).
    Empty input selects nothing. Re-prompts until the input parses.
    """

    print(format_msg(PROMPT_COLOUR, True, header))
    max_idx_w = len(str(len(items)))
    for i, item in enumerate(items):
        print(format_msg(PROMPT_COLOUR, True, f"  {i + 1:<{max_idx_w}}\t{item}"))
    print(format_msg(PROMPT_COLOUR, True, "[A]ll or (1 2 3, 1-3, ^4)"))

    def valid_idx(v: str) -> int:
        try:
            idx = int(v, base=10) - 1  # -1 to translate to 0 index
        except ValueError:
            raise ValueError(f'Given value "{v}" must be an integer.')
        if idx < 0 or idx >= len(items):
            raise ValueError(f'Given value "{v}" must be between 1 and {len(items)} inclusive.')
        return idx

    def parse(ans: str) -> list[str]:
        if ans in ("a", "all"):
            return list(items)
        if not ans:
            return []

        selected: list[str] = []
        for tok in ans.split():
            fr, sep, to = tok.partition("-")
            if sep:
                lo, hi = valid_idx(fr), valid_idx(to)
                if lo > hi:
                    raise ValueError(f'Given range "{tok}" must be lo-hi.')
                selected += items[lo : hi + 1]
            elif tok.startswith("^"):
                t = valid_idx(tok[1:])
                selected += items[:t] + items[t + 1 :]
            else:
                selected.append(items[valid_idx(tok)])
        return list(set(selected))

    while True:
        ans = prompt("", end="").lower().strip()
        try:
            return parse(ans)
        except ValueError as e:
            warn(f"invalid input. {e} Please try again.")


def pause() -> None:
    if _disable_input:
        return

    _input("\n\033[2m\033[3m(Ctrl+C to exit, enter to continue)\033[0m")
    print("\033[1A\r\033[2K\033[1A\r\033[2K", end="")  # Clear pause prompt

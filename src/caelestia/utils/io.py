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
    """Log exceptions to stdout instead of raising

    Used by the `apply_()` functions so that an exception, when applying
    a theme, does not prevent the other themes from being applied.
    """

    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            error(f'exception during "{func.__name__}()": {str(e)}')

    return wrapper


def format_msg(colour: int, msg: str) -> str:
    return f"\033[{colour}m:: {msg}\033[0m"


def log(msg: str) -> None:
    print(format_msg(LOG_COLOUR, msg))


def info(msg: str) -> None:
    print(format_msg(INFO_COLOUR, msg))


def warn(msg: str) -> None:
    print(format_msg(WARNING_COLOUR, f"Warning: {msg}"))


def error(err: str | Exception) -> None:
    print(format_msg(ERROR_COLOUR, f"Error: {err}"), file=sys.stderr)


def fatal(err: str | Exception) -> Never:
    print(format_msg(ERROR_COLOUR, f"Fatal: {err}"), file=sys.stderr)
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


def prompt(msg: str, end: str = " ") -> str:
    return _input(format_msg(PROMPT_COLOUR, msg) + end)


def confirm(msg: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    answer = prompt(msg + suffix).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def pause() -> None:
    if _disable_input:
        return

    _input("\n\033[2m\033[3m(Ctrl+C to exit, enter to continue)\033[0m")
    print("\033[1A\r\033[2K\033[1A\r\033[2K", end="")  # Clear pause prompt

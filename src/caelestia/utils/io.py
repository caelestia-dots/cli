import sys
from typing import Never


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


def _format_msg(colour: int, msg: str) -> str:
    return f"\033[{colour}m:: {msg}\033[0m"


def log(msg: str) -> None:
    print(_format_msg(2, msg))


def info(msg: str) -> None:
    print(_format_msg(0, msg))


def warn(msg: str) -> None:
    print(_format_msg(33, f"Warning: {msg}"))


def error(err: str | Exception) -> None:
    print(_format_msg(31, f"Error: {err}"), file=sys.stderr)


def fatal(err: str | Exception) -> Never:
    print(_format_msg(31, f"Fatal: {err}"), file=sys.stderr)
    sys.exit(1)


def _input(prompt: str) -> str:
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        print()
        raise KeyboardInterrupt()


def prompt(msg: str) -> str:
    return _input(_format_msg(36, msg) + " ")


def pause() -> None:
    _input("\n\033[2m\033[3m(Ctrl+C to exit, enter to continue)\033[0m")
    print("\033[1A\r\033[2K\033[1A\r\033[2K", end="")  # Clear pause prompt

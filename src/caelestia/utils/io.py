import sys
from time import strftime


def log_message(message: str) -> None:
    timestamp = strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def log_exception(func):
    """Log exceptions to stdout instead of raising

    Used by the `apply_()` functions so that an exception, when applying
    a theme, does not prevent the other themes from being applied.
    """

    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            log_message(f'Error during execution of "{func.__name__}()": {str(e)}')

    return wrapper


def _format_msg(colour: int, msg: str) -> str:
    return f"\033[{colour}m:: {msg}\033[0m"


def log(msg: str) -> None:
    print(_format_msg(2, msg))


def info(msg: str) -> None:
    print(_format_msg(0, msg))


def warn(msg: str) -> None:
    print(_format_msg(33, f"Warning: {msg}"))


def error(msg: str) -> None:
    print(_format_msg(31, f"Error: {msg}"), file=sys.stderr)


def fatal(msg: str) -> None:
    print(_format_msg(31, f"Fatal: {msg}"), file=sys.stderr)
    sys.exit(1)


def prompt(msg: str) -> str:
    return input(_format_msg(36, msg) + " ")


def pause() -> None:
    input("\033[2m\033[3m(Ctrl+C to exit, enter to continue)\033[0m")
    print("\033[1A\r\033[2K", end="")  # Clear pause prompt

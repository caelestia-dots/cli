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

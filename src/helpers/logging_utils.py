# src/helpers/logging_utils.py
import logging
import time
import functools
import inspect
from logging.handlers import RotatingFileHandler

def setup_logging(
    name: str = "zoom_app",
    level: int = logging.DEBUG,
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    log_file: str | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure logging once and return the root app logger.
    If log_file is provided, add a rotating file handler.
    """
    logger = logging.getLogger(name)
    if getattr(setup_logging, "_configured", False):
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file:
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    setup_logging._configured = True
    return logger  # standard logger usage per docs [web:39][web:23]

def get_logger(name: str = "zoom_app") -> logging.Logger:
    return logging.getLogger(name)  # canonical accessor [web:39]

def _short(value, maxlen: int = 200) -> str:
    s = repr(value)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"

def trace_calls(logger: logging.Logger | None = None, level: int = logging.DEBUG):
    """
    Decorator factory that logs entry, args/kwargs, exit with duration, and exceptions.
    Works for sync and async functions.
    """
    def decorate(func):
        use_log = logger or logging.getLogger(f"trace.{func.__module__}")
        is_coroutine = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = f"{func.__module__}.{func.__qualname__}"
            args_repr = ", ".join([_short(a) for a in args] + [f"{k}={_short(v)}" for k, v in kwargs.items()])
            use_log.log(level, f"→ enter {name}({args_repr})")
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                dt = (time.perf_counter() - t0) * 1000.0
                use_log.log(level, f"← exit  {name} [{dt:.2f} ms] → { _short(result) }")
                return result
            except Exception:
                dt = (time.perf_counter() - t0) * 1000.0
                use_log.exception(f"✖ error {name} [{dt:.2f} ms]")
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = f"{func.__module__}.{func.__qualname__}"
            args_repr = ", ".join([_short(a) for a in args] + [f"{k}={_short(v)}" for k, v in kwargs.items()])
            use_log.log(level, f"→ enter {name}({args_repr})")
            t0 = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                dt = (time.perf_counter() - t0) * 1000.0
                use_log.log(level, f"← exit  {name} [{dt:.2f} ms] → { _short(result) }")
                return result
            except Exception:
                dt = (time.perf_counter() - t0) * 1000.0
                use_log.exception(f"✖ error {name} [{dt:.2f} ms]")
                raise

        return async_wrapper if is_coroutine else sync_wrapper
    return decorate  # decorator approach per cookbook and best practices [web:20][web:23]

def trace_class(cls=None, *, logger: logging.Logger | None = None, level: int = logging.DEBUG):
    """
    Class decorator that applies trace_calls to all public methods.
    Usage:
      @trace_class
      class C: ...
    or:
      @trace_class(logger=get_logger(), level=logging.DEBUG)
      class C: ...
    """
    def wrap(target_cls):
        for name, attr in list(vars(target_cls).items()):
            if callable(attr) and not name.startswith("_"):
                setattr(target_cls, name, trace_calls(logger=logger, level=level)(attr))
        return target_cls

    if cls is None:
        return wrap
    return wrap(cls)

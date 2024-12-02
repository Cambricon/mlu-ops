"""
    init logging module by input argument
"""

__all__ = (
    "logger_init",
)

import logging
import functools

def logger_init(args):
    logging.getLogger().setLevel(args.logging_level)
    logging.basicConfig(
        format='%(levelname)s:%(name)s [%(asctime)s.%(msecs)03d] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S',
    )

# use decorator to print logs at the entry of a function
def log_debug(func):
    filename = func.__code__.co_filename
    lineno = func.__code__.co_firstlineno + 1
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug("[{}: {}] {} start.".format(filename, lineno, func.__name__))
        ret = func(*args, **kwargs)
        logging.debug("[{}: {}] {} end.".format(filename, lineno, func.__name__))
        return ret
    return wrapper

def log_info(func):
    filename = func.__code__.co_filename
    lineno = func.__code__.co_firstlineno + 1
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info("[{}: {}] {} start.".format(filename, lineno, func.__name__))
        ret = func(*args, **kwargs)
        logging.info("[{}: {}] {} end.".format(filename, lineno, func.__name__))
        return ret
    return wrapper

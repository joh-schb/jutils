import os
import sys
import logging
import datetime
import functools
from termcolor import colored


__all__ = [
    "get_logger",
]
# ===============================================================================================


@functools.lru_cache()
def get_logger(log_dir='./logs', rank=0, prefix=None, log_to_file=True, log_to_stdout=True):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.datetime.now().strftime('%y%m%d-%H%M%S')

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # ensure that the logger has at least one handler
    logger.addHandler(logging.NullHandler())

    # log to file
    if log_to_file:
        prefix = prefix + "_" if prefix is not None else ""
        filename = f'log_rank{rank}_{prefix}{timestamp}.txt'
        filepath = os.path.join(log_dir, filename)

        fh = logging.FileHandler(filepath)
        fh.setLevel(logging.INFO)
        # fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
        fmt = '[%(asctime)s] %(levelname)s : %(message)s'
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))

        logger.addHandler(fh)

    # log to stdout
    if log_to_stdout:
        color_fmt = colored('[%(asctime)s]', 'green') + \
                    ' %(levelname)s : %(message)s'
                    # colored('(%(filename)s %(lineno)d)', 'yellow') + \

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))

        logger.addHandler(sh)

    return logger


if __name__ == "__main__":
    my_logger = get_logger(log_to_file=True, log_to_stdout=True)
    my_logger.info("Hello world!")
    my_logger.warning("This is a warning!")

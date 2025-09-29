import time
import datetime
from typing import Union


__all__ = [
    "format_time",
    "timer",
    "get_time",
    "Timer",
]
# ===============================================================================================


def format_time(seconds: Union[int, float]) -> str:
    """ Convert seconds to human readable string with hours, minutes and seconds. """
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def timer(start, end):
    return format_time(end - start)


def get_time(str_format: str = '%H:%M:%S.%f'):
    return datetime.datetime.now().strftime(str_format)


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.delta = self.end - self.start
        self.time = format_time(self.delta)


if __name__ == "__main__":
    # format_time
    print("format_time(123456):", format_time(90123.45))

    # timer(start, end)
    t0 = time.time()
    time.sleep(2)
    print("timer(start, end):", timer(t0, time.time()))

    # Timer
    with Timer() as t:
        time.sleep(2)
    print("Timer():", t.time)

    # timing
    print("get_time():", get_time())

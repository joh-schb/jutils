import time
import datetime


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def get_time(str_format: str = '%H:%M:%S.%f'):
    return datetime.datetime.now().strftime(str_format)


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.time = timer(self.start, self.end)


if __name__ == "__main__":
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

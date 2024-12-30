import os
import sys
import math
from typing import Any
from contextlib import contextmanager


def exists(x):
    return x is not None


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def bool2str(val):
    return "True" if val else "False"


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


class Namespace(dict):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def pad_v_like_x(v_, x_):
    """
    Function to reshape the vector by the number of dimensions
    of x. E.g. x (bs, c, h, w), v (bs) -> v (bs, 1, 1, 1).
    """
    if isinstance(v_, float):
        return v_
    return v_.reshape(-1, *([1] * (x_.ndim - 1)))


class NullObject:
    """ just do nothing """
    def __getattr__(self, name) -> "NullObject":
        return NullObject()

    def __call__(self, *args: Any, **kwds: Any) -> "NullObject":
        return NullObject()
    
    def __enter__(self) -> "NullObject":
        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


if __name__ == "__main__":
    # convert_size(size_bytes)
    print("convert_size(1_000_000_000):", convert_size(1_000_000_000))

    # Namespace()
    self = Namespace(a=1)
    self.b = "variable b"
    print("Namespace:", self.a, self.b)

    # suppress
    with suppress_stdout():
        print("This will not be printed.")
    print("This will be printed.")

    # pad_vector_like_x(v_, x_)
    import torch
    print("pad_vector_like_x(v_, x_):", pad_v_like_x(torch.randn(10), torch.randn(10, 3, 224, 224)).shape)
    import numpy as np
    print("pad_vector_like_x(v_, x_):", pad_v_like_x(np.random.randn(10), np.random.randn(10, 3, 224, 224)).shape)

    # NullObject
    null = NullObject()
    print("NullObject():", null('a', 123))

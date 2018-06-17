"""This module provides the functionality to generate interesting
1-dimensional and 2-dimensional signals for the purposes of analysis."""


import numpy as np
import random
import itertools


def random_1d_signal(m, n):
    """Return a random 1-dimensional signal comprised of 'n' cosines of
    length 'm' with amplitude between 1 and 10 and frequency between
    2*pi*1 and 2*pi*10.
    """
    amp = 10
    freq = 10
    s = random.randint(1, amp)*np.cos(np.linspace(0, 2*np.pi*random.randint(1, freq), m))
    for _ in range(n - 1):
        s = s + random.randint(1, amp)*np.cos(np.linspace(0, 2*np.pi*random.randint(1, freq), m))
    return s


def square_1d_signal(m):
    """Return a 1-dimensional square wave of length 'm'.
    """
    s = np.zeros(m)
    s[int(m/4):int(3*m/4)] = 1
    return s


def chequered(m, n, k):
    """Return a 2-dimensional chequered matrix of dimension (m by n).
    """
    array = np.zeros([m, n], dtype=np.uint8)
    for x, y in itertools.product(range(n), range(m)):
        if (x % k) // (k//2) == (y % k) // (k//2):
            array[y, x] = 0
        else:
            array[y, x] = 255
    return array
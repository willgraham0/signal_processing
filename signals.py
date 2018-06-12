"""This module provides the functionality to generate interesting
1-dimensional and 2-dimensional signals for the purposes of analysis."""


import numpy as np
import random


def random_1d_signal(n, l):
    """Return a random 1d signal comprised of 'n' cosines of length 'l'
    with amplitude between 1 and 10.
    """
    amp = 10
    s = random.randint(1, amp)*np.cos(np.linspace(0, 2*np.pi*random.randint(0, l), l))
    for _ in range(n - 1):
        s = s + random.randint(1, amp)*np.cos(np.linspace(0, 2*np.pi*random.randint(0, l), l))
    return s


def square_1d_signal(l):
    """Return a square wave of overall length 'l'.
    """
    s = np.zeros(l)
    s[int(l/4):int(3*l/4)] = 1
    return s

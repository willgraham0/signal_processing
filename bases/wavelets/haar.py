"""This module provides the functionality to transform 1-dimensional
and 2-dimenional signals into the Haar wavelet domain and back into the
functional domain."""


import numpy as np


def haar_matrix(m):
    """Return the Haar wavelet matrix (m by m) where log2(m) is a real
    number. This matrix multiplies a vector of coefficients to construct
    a signal.
    """
    if not np.log2(n).is_integer():
        raise ValueError("The value of log2(m) must be a whole number.")
    vT = np.ones

def square(k, m):
    x = np.sin(np.linspace(0, 2*np.pi, k))
    s = np.piecewise(x, [x < 0, x >= 0], [-1, 1])
    return np.append(s, np.zeros(m-k))
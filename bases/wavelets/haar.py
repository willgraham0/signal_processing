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
    number_of_wavelets = m - 1




def haar_wavelet(k, m):
    """Return a Haar wavelet of length k spanning the signal length m,
    where k <= m. The values of log2(k) and log2(m) must be integers. 
    E.g. square(2, 8) -> np.array([1, -1,  0,  0,  0,  0,  0,  0])
    E.g. square(4, 8) -> np.array([1,  1, -1, -1,  0,  0,  0,  0])
    """
    if not np.log2(k).is_integer() and not np.log(m).is_integer():
        raise ValueError("The values of log2(k) and log2(m) must be whole numbers.")
    x = np.sin(np.linspace(0, 2*np.pi, k))
    s = np.piecewise(x, [x < 0, x >= 0], [-1, 1])
    return np.append(s, np.zeros(m - k))
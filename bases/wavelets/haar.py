"""This module provides the functionality to transform 1-dimensional
and 2-dimenional signals into the Haar wavelet domain and back into the
functional domain.
"""

import numpy as np


def haar_wavelet(k, m):
    """Return a Haar wavelet of length k spanning the signal length m,
    where k <= m. The values of log2(k) and log2(m) must be integers. 
    E.g. haar_wavelet(2, 8) -> np.array([1, -1,  0,  0,  0,  0,  0,  0])
    E.g. haar_wavelet(4, 8) -> np.array([1,  1, -1, -1,  0,  0,  0,  0])
    """
    if k > m:
        raise ValueError("The vale of k must be less than or equal to m.")
    if not np.log2(k).is_integer() or not np.log2(m).is_integer():
        raise ValueError("The values of log2(k) and log2(m) must be whole numbers.")
    x = np.sin(np.linspace(0, 2*np.pi, k))
    s = np.piecewise(x, [x < 0, x >= 0], [-1, 1])
    return np.append(s, np.zeros(m - k))


def haar_matrix(m):
    """Return the Haar wavelet matrix (m by m) where log2(m) is a real
    number. This matrix multiplies a vector of coefficients to construct
    a signal.
    """
    if not np.log2(m).is_integer():
        raise ValueError("The value of log2(m) must be a whole number.")
    vectors = []
    for i in map(lambda x: 2**x, reversed(range(int(np.log2(2*m))))):
        if i == 1:
            vectors = [np.ones(m)] + vectors
        else:
            wavelet = haar_wavelet(i, m)
            vectors.append(wavelet)
            for _ in range(int(m / i) - 1):
                wavelet = np.roll(wavelet, i)
                vectors.append(wavelet)
    return np.column_stack(vectors)

"""This module provides the functionality to transform 1-dimensional
and 2-dimenional signals into the Haar wavelet domain and back into the
functional domain."""


import numpy as np


def haar_matrix(m):
    """Return the Haar wavelet matrix (m by m) where m is an even
    number. This matrix multiplies a vector of coefficients to construct
    a signal.
    """
    if m // 2 ! = 0:
        raise ValueError
    


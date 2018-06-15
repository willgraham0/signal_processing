"""This module provides the functionality to transform 1-dimensional
and 2-dimenional signals into the wavelet domain and back into the
functional domain.
"""

from .families import Wavelets, Haar
import numpy as np


def is_implemented(family):
    """Return True if the wavelet family is implemented. Else False.
    """
    if family in Wavelets.families.keys():
        return True
    else:
        return False


def get_family(family):
    """Return the family class object. Else raise a NotImplementedError.
    """
    if is_implemented(family):
        return Wavelets.families[family]
    else:
        raise NotImplementedError('{} family is not implemented.'.format(family))


def dwt(signal, family):
    """Return the Wavelet coefficients of a 1-dimensional signal.
    """
    Family = get_family(family)
    return Family.imatrix(len(signal)).dot(signal)
        

def idwt(coefficients, family):
    """Return the 1-dimensional signal from its Wavelet coefficients.
    """
    Family = get_family(family)
    return Family.matrix(len(coefficients)).dot(coefficients)


def heatmap_matrix(signal, family):
    """Return a 2-dimensional array of the wavelet matrix, with each
    wavelet scaled by its corresponding coefficient (its amplitude) and
    compressed (so that the number of columns equals the number of
    dilations) for a 1-dimensional signal."""
    Family = get_family(family)
    wavelet_matrix = Family.matrix(len(signal))
    coefficients = dwt(signal, family)
    pre_compress = np.multiply(wavelet_matrix, coefficients)
    return Family.compress(pre_compress)



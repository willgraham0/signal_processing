"""This module provides the functionality to transform 1-dimensional
and 2-dimenional signals into the wavelet domain and back into the
functional domain.
"""

from .families import Wavelets, Haar


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


# def dwt(signal, family):
#     if is_implemented(family):
        

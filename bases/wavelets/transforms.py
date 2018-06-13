"""This module provides the functionality to transform 1-dimensional
and 2-dimenional signals into the wavelet domain and back into the
functional domain.
"""

from .families import Wavelets, Haar


def is_provisioned(wavelet_family_name):
    """Return True if the wavelet family is provisioned. Else False.
    """
    if wavelet_family_name in [name for name, cls in Wavelets.families]:
        return True
    else:
        return False

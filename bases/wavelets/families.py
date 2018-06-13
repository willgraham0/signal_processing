"""This module provides the definitions of the different families of
wavelets.
"""

import numpy as np


class Wavelets(type):
    """A metaclass to encapsulate all the families of wavelets.
    """
    
    families = []
    def __init__(cls, name, bases, attrs):
        """Upon this metaclass being used, add the new class' name and
        the class itself to the metaclass variable 'families'.
        """
        Wavelets.families.append((name, cls))


class Haar(metaclass=Wavelets):
    """The Haar family of wavelets.
    """

    @staticmethod
    def wavelet(k, m):
        """Return a normalised Haar wavelet of length k spanning the signal 
        length m, where k <= m. The values of log2(k) and log2(m) must be
        integers. E.g: 
        Haar.wavelet(2, 8) -> 1/np.sqrt(2)*np.array([1, -1,  0,  0,  0,  0,  0,  0])
        Haar.wavelet(4, 8) -> 1/np.sqrt(4)*np.array([1,  1, -1, -1,  0,  0,  0,  0])
        """
        if k > m:
            raise ValueError("The vale of k must be less than or equal to m.")
        if not np.log2(k).is_integer() or not np.log2(m).is_integer():
            raise ValueError("The values of log2(k) and log2(m) must be whole numbers.")
        if k == 1:
            return (1/np.sqrt(m))*np.ones(m)
        else:
            x = np.sin(np.linspace(0, 2*np.pi, k))
            s = np.piecewise(x, [x < 0, x >= 0], [-1, 1])
            return (1/np.sqrt(k))*np.append(s, np.zeros(m - k))


    @staticmethod
    def matrix(m):
        """Return the Haar wavelet matrix (m by m) where log2(m) is a real
        number. This matrix multiplies a vector of coefficients to construct
        a signal.
        """
        if not np.log2(m).is_integer():
            raise ValueError("The value of log2(m) must be a whole number.")
        vectors = []
        for i in map(lambda x: 2**x, reversed(range(int(np.log2(2*m))))):
            wavelet = Haar.wavelet(i, m)
            if i == 1:
                vectors = [wavelet] + vectors
            else:
                vectors.append(wavelet)
                for _ in range(int(m / i) - 1):
                    wavelet = np.roll(wavelet, i)
                    vectors.append(wavelet)
        return np.column_stack(vectors)


    @staticmethod
    def imatrix(m):
        """Return the inverse Haar wavelet matrix (m by m) where log2(m) is
        a real number. This matrix multiplies a signal to obtain a vector of
        coefficients a signal.
        """
        return Haar.matrix(m).T

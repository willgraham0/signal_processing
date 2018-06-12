"""This module provides the functionality to transform 1-dimensional
and 2-dimenional signals into the Fourier domain and back into the
functional domain."""


import numpy as np


def dft(signal):
    """Return the Fourier coefficients of a 1-dimensional signal.
    """
    return dft_matrix(len(signal)).dot(signal)


def idft(coefficients):
    """Return the 1-dimensional signal from its Fourier coefficients.
    """
    return idft_matrix(len(coefficients)).dot(coefficients)


def dft2(signal):
    """Return the Fourier coefficients of a 2-dimensional signal.
    """
    rows, cols = signal.shape
    return dft_matrix(rows).dot(signal.dot(dft_matrix(cols)))


def idft2(coefficients):
    """Return the 2-dimensional signal from its Fourier coefficients."""
    rows, cols = coefficients.shape
    return idft_matrix(rows).dot(coefficients.dot(idft_matrix(cols)))


def dft_matrix(m):
    """Return the discrete Fourier transform matrix (m by m). This
    matrix multiplies a signal to obtain a vector of coefficients.
    """    
    k, j = np.meshgrid(np.arange(m), np.arange(m))
    omega = np.exp(-2*np.pi*1j/m)
    return np.power(omega, k*j)/np.sqrt(m)


def idft_matrix(m):
    """Return the inverse discrete Fourier transform matrix (m by m).
    This matrix multiplies a vector of coefficients to construct a
    signal.
    """
    k, j = np.meshgrid(np.arange(m), np.arange(m))
    omega = np.exp(2*np.pi*1j/m)
    return np.power(omega, k*j)/np.sqrt(m)

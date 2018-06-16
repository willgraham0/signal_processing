"""This module provides the functionality to transform 1-dimensional
and 2-dimenional signals into the Fourier domain and back into the
functional domain.
"""


import numpy as np


def dft(signal):
    """Return the Fourier coefficients of a 1-dimensional signal.
    """
    return ifourier_matrix(len(signal)).dot(signal)


def fft(signal):
    """Return the Fourier coefficients of a 1-dimensional signal using
    the Fast Fourier Transform algorithm.
    TO BE IMPLEMENTED"""
    pass


def idft(coefficients):
    """Return the 1-dimensional signal from its Fourier coefficients.
    """
    return fourier_matrix(len(coefficients)).dot(coefficients)


def ifft(coefficients):
    """Return the 1-dimensional signal from its Fourier coefficients
    using the Fast Fourier Transform algorithm.
    TO BE IMPLEMENTED"""
    pass


def dft2(signal):
    """Return the Fourier coefficients of a 2-dimensional signal.
    """
    rows, cols = signal.shape
    return ifourier_matrix(rows).dot(signal.dot(ifourier_matrix(cols)))


def idft2(coefficients):
    """Return the 2-dimensional signal from its Fourier coefficients."""
    rows, cols = coefficients.shape
    return fourier_matrix(rows).dot(coefficients.dot(fourier_matrix(cols)))


def fourier_matrix(m):
    """Return the orthonormal Fourier matrix (m by m). This matrix
    multiplies a vector of coefficients to construct a signal.
    """
    k, j = np.meshgrid(np.arange(m), np.arange(m))
    omega = np.exp(2*np.pi*1j/m)
    return np.power(omega, k*j)/np.sqrt(m)


def ifourier_matrix(m):
    """Return the inverse orthonormal Fourier matrix (m by m). This
    matrix multiplies a signal to obtain a vector of coefficients.
    """    
    k, j = np.meshgrid(np.arange(m), np.arange(m))
    omega = np.exp(-2*np.pi*1j/m)
    return np.power(omega, k*j)/np.sqrt(m)

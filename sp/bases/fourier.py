"""This module provides the functionality to transform 1-dimensional
and 2-dimenional signals into the Fourier domain and back into the
functional domain.
"""


import numpy as np


def idft(signal):
    """Return the Fourier coefficients of a 1-dimensional signal.
    """
    shape = signal.shape
    if len(shape) != 1:
        raise ValueError('Signal is not 1-dimensional.')
    return ifourier_matrix(len(signal)).dot(signal)


def dft(coefficients):
    """Return the 1-dimensional signal from its Fourier coefficients.
    """
    shape = coefficients.shape
    if len(shape) != 1:
        raise ValueError('Signal is not 1-dimensional.')
    return fourier_matrix(len(coefficients)).dot(coefficients)


def fft(coefficients):
    """Return the 1-dimensional signal from its Fourier coefficients
    using the Fast Fourier Transform algorithm.
    TO BE IMPLEMENTED"""
    pass


def idft2(signal):
    """Return the Fourier coefficients of a 2-dimensional signal.
    """
    shape = signal.shape
    if len(shape) != 2:
        raise ValueError('Signal is not 2-dimensional.')
    rows, cols = shape
    return ifourier_matrix(rows).dot(signal.dot(ifourier_matrix(cols)))


def dft2(coefficients):
    """Return the 2-dimensional signal from its Fourier coefficients."""
    shape = coefficients.shape
    if len(shape) != 2:
        raise ValueError('Signal is not 2-dimensional.')
    rows, cols = shape
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


def diagonal(m):
    """Return the diagonal matrix of the Fourier matrix.
    """
    j = np.arange(m)
    omega = np.exp(-2*np.pi*1j/m)
    return np.diag(np.power(omega, j))


def idiagonal(m):
    """Return the diagonal matrix of the inverse Fourier matrix.
    """
    j = np.arange(m)
    omega = np.exp(2*np.pi*1j/m)
    return np.diag(np.power(omega, j))

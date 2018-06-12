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


def dft2(image):
    """Return the Fourier coefficients of a 2-dimensional signal.
    """
    rows, cols = image.shape
    return dft_matrix(rows).dot(image.dot(dft_matrix(cols)))


def idft2(coefficients):
    """Return the 2-dimensional signal from its Fourier coefficients."""
    rows, cols = coefficients.shape
    return idft_matrix(rows).dot(coefficients.dot(idft_matrix(cols)))


def dft_matrix(M):
    """Return the discrete Fourier transform matrix. This matrix
    multiplies a signal to obtain a vector of coefficients.
    """    
    k, j = np.meshgrid(np.arange(M), np.arange(M))
    omega = np.exp(-2*np.pi*1j/M)
    return np.power(omega, k*j)/np.sqrt(M)


def idft_matrix(M):
    """Return the inverse discrete Fourier transform matrix. This matrix
    multiplies a vector of coefficients to construct a signal.
    """
    k, j = np.meshgrid(np.arange(M), np.arange(M))
    omega = np.exp(2*np.pi*1j/M)
    return np.power(omega, k*j)/np.sqrt(M)

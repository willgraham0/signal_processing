import numpy as np
import matplotlib.pyplot as plt
import random


def coroutine(func):
    """A coroutine decorator for calling the initial 'next' function.
    """
    def start(*args, **kwargs):
        g = func(*args, **kwargs)
        next(g)
        return g
    return start


def random_1d_signal(n, l):
    """Return a random 1d signal comprised of 'n' cosines of length 'l'
    with amplitude between 1 and 10.
    """
    amp = 10
    s = random.randint(1, amp)*np.cos(np.linspace(0, 2*np.pi*random.randint(0, l), l))
    for _ in range(n - 1):
        s = s + random.randint(1, amp)*np.cos(np.linspace(0, 2*np.pi*random.randint(0, l), l))
    return s


def square_1d_signal(l):
    """Return a square wave of overall length 'l'.
    """
    s = np.zeros(l)
    s[int(l/4):int(3*l/4)] = 1
    return s


def attenuate(signal, restore=False):
    """Return the linearly attenuated signal.
    """
    down = np.linspace(1, 0.1, int(len(signal)/2))
    up   = np.linspace(0.1, 1, len(signal)-int(len(signal)/2))
    attenuate = np.concatenate([down, up])
    if restore:
        return signal/attenuate
    return signal*attenuate


@coroutine
def plot(signal):
    """Plot the signal. New signals can be sent to this coroutine.
    """
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    while True:
        if signal.dtype == np.complex128:
            signal = signal.real
        ax.plot(np.arange(len(signal)), signal)
        fig.canvas.draw()
        signal = (yield)


def idft(coefficients):
    """Return the signal from its Fourier coefficients.
    """
    return idft_matrix(len(coefficients)).dot(coefficients)


def dft(signal):
    """Return the Fourier coefficients of the signal.
    """
    return dft_matrix(len(signal)).dot(signal)


def idft_matrix(M):
    """Return the inverse discrete Fourier transform matrix. This matrix
    multiplies a vector of coefficients to construct a signal.
    """
    k, j = np.meshgrid(np.arange(M), np.arange(M))
    omega = np.exp(2*np.pi*1j/M)
    return np.power(omega, k*j)/np.sqrt(M)


def dft_matrix(M):
    """Return the discrete Fourier transform matrix. This matrix
    multiplies a signal to obtain a vector of coefficients.
    """    
    k, j = np.meshgrid(np.arange(M), np.arange(M))
    omega = np.exp(-2*np.pi*1j/M)
    return np.power(omega, k*j)/np.sqrt(M)

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


# @coroutine
# def plot(signal):
#     """Plot the signal. New signals can be sent to this coroutine.
#     """
#     plt.ion()
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     while True:
#         if signal.dtype == np.complex128:
#             signal = signal.real
#         ax.plot(np.arange(len(signal)), signal)
#         fig.canvas.draw()
#         signal = (yield)


@coroutine
def plot(signal):
    """Plot the signal. New signals can be sent to this coroutine.
    """
    plt.ion()
    fig = plt.figure()
    dimensions = len(signal.shape)
    ax = fig.add_subplot(1, 1, 1)
    while True:
        if len(signal.shape) == dimensions:
            if signal.dtype == np.complex128:
                signal = signal.real
            if dimensions == 1:
                ax.plot(np.arange(len(signal)), signal)
            if dimensions == 2:
                n = len(fig.axes)
                if n > 1:
                    for i in range(n):
                        fig.axes[i].change_geometry(n+1, 1, i+1)
                    ax = fig.add_subplot(n+1, 1, n+1)
                ax.imshow(signal)
            fig.canvas.draw()
        else:
            print("You're trying to plot a {}-dimensional signal on a plot for a {}-dimensional signal.".format(len(signal.shape), dimensions))
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


def dft2(image):
    """Return the 2d Fourier coefficients of the signal.
    """
    rows, cols = image.shape
    return dft_matrix(rows).dot(image.dot(dft_matrix(cols)))

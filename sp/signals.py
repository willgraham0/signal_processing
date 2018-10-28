"""This module provides the functionality to generate interesting
1-dimensional and 2-dimensional signals for the purposes of analysis."""


import numpy as np
import itertools


def sum_of_sinusoids(m, amps_freqs):
    """Return a 1-dimensional signal comprised of a sum of cosines.
    The variable amps_freqs is a list of lists of amplitude and frequency
    (in radians) for each cosine. Therefore the length of amsp_freqs is 
    the number of cosines being summed.
    """
    s = np.zeros(m)
    for amp, freq in amps_freqs:
        s = s + amp*np.cos(2*np.pi*freq*1/m*np.arange(0, m))
    return s


def square_signal(m):
    """Return a 1-dimensional square wave of length 'm'.
    """
    s = np.zeros(m)
    s[int(m/4):int(3*m/4)] = 1
    return s


def chequered(m, n, t):
    """Return a 2-dimensional array (m by n) of black and white
    chequers of thickness, t.
    """
    s = np.zeros((m, n))
    mchunks = list(chunks(range(m), t))
    nchunks = list(chunks(range(n), t))
    for i, mchunk in enumerate(mchunks):
        for j, nchunk in enumerate(nchunks):
            if (i%2 == 0 and j%2 != 0) or (i%2 != 0 and j%2 == 0):
                s[mchunk[0]:mchunk[-1]+1, nchunk[0]:nchunk[-1]+1] = 255
    return s


def stripes(m, n, t, vertical=True):
    """Return a 2-dimensional array of black and white stripes, either
    horizontal or vertical.
    """
    s = np.zeros((m, n))
    flag = True
    if vertical:
        for i, chunk in enumerate(chunks(range(n), t)):
            if i//2 == 0:
                s[:, chunk] = 255
    else:
        for i, chunk in enumerate(chunks(range(m), t)):
            if i//2 == 0:
                s[chunk, :] = 255
    return s


def sinusoids_2d(m, n, f):
    """Return a 2-dimensional array (m by n) of a sinusoid of amplitude
    of 1, frequency of f (in radians) and length of m, repeated n number
    of times.
    """
    s = np.cos(2*np.pi*f*1/m*np.arange(0, m))   
    return np.tile(s, (n, 1)).T


def vertical_sinusoids(m, n, f):
    """Return a 2-dimensional array (m by n) of a sinusoid of amplitude
    of 1, frequency of f (in radians) and length of m, repeated n number
    of times.
    """
    return sinusoids_2d(m, n, f)


def horizontal_sinusoids(m, n, f):
    """Return a 2-dimensional array (m by n) of a sinusoid of amplitude
    of 1, frequency of f (in radians) and length of m, repeated n number
    of times.
    """
    return sinusoids_2d(n, m, f).T


def chunks(a_list, n):
    """Return ranges of length 'n' along the list provided.
    """
    for i in range(0, len(a_list), n):
        yield a_list[i: i+n]

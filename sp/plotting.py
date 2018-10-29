"""This module provides functionality for plotting signals and their
transformations into other bases.
"""


import itertools

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

from . import bases


def coroutine(func):
    """A coroutine decorator for calling the initial 'next' function
    automatically.
    """
    def start(*args, **kwargs):
        g = func(*args, **kwargs)
        next(g)
        return g
    return start


@coroutine
def plot(signal, grid=False):
    """Plot the signal. New signals can be sent to this coroutine via
    the coroutine's send method.
    """
    plt.ion()
    fig = plt.figure()
    dimensions = len(signal.shape)
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(grid)
    first = True
    while True:
        if len(signal.shape) == dimensions:
            if signal.dtype == np.complex128:
                signal = signal.real
            if dimensions == 1:
                ax.set_xlabel('Dimension')
                ax.set_ylabel('Value')
                ax.plot(np.arange(len(signal)), signal)
            if dimensions == 2:
                if first:
                    first = False
                else:
                    n = len(fig.axes)
                    for i in range(n):
                        fig.axes[i].change_geometry(1, n+1, i+1)
                    ax = fig.add_subplot(1, n+1, n+1)
                ax.imshow(signal, cmap='Greys')
            fig.canvas.draw()
        else:
            print("You're trying to plot a {}-dimensional signal on a plot for a {}-dimensional signal.".format(len(signal.shape), dimensions))
        signal = (yield)


def plot_wavelet_heatmap(signal, family):
    """Plot a heatmap of amplitudes of wavelets for each wavelet
    dilation across length of signal.
    """
    matrix = bases.wavelets.heatmap_matrix(signal, family)
    m, n = matrix.shape
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='Greys')
    ax.set_title('Amplitudes with Dimensions versus Dilation/Frequency')
    ax.set_xlabel('Dilation/Frequency')
    ax.set_ylabel('Dimension')
    for i, j in itertools.product(range(m), range(n)):
        text = ax.text(j, i, round(matrix[i, j], 2), ha="center", va="center", color="w")
        text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                               path_effects.Normal()])
    fig.tight_layout()
    plt.show()
    
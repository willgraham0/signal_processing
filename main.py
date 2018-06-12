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



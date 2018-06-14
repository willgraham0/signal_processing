"""This module provides functionality for plotting signals and their
transformations into other bases."""


import matplotlib.pyplot as plt


def coroutine(func):
    """A coroutine decorator for calling the initial 'next' function
    automatically.
    """
    def start(*args, **kwargs):
        g = func(*args, **kwargs)
        next(g)
        return g
    return start


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
    """Plot the signal. New signals can be sent to this coroutine via
    the coroutine's send method.
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

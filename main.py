import numpy as np

import random







def attenuate(signal, restore=False):
    """Return the linearly attenuated signal.
    """
    down = np.linspace(1, 0.1, int(len(signal)/2))
    up   = np.linspace(0.1, 1, len(signal)-int(len(signal)/2))
    attenuate = np.concatenate([down, up])
    if restore:
        return signal/attenuate
    return signal*attenuate





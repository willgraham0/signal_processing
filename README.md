# Signal Processing

## Introduction

Any signal can be thought of as a _vector_ that lies in a _vector space_.
The vector can be formed by the linear combination of other _indepedent_
vectors that lie in the same vector space. The independent vectors are
ones that cannot be formed by the linear combination of the others.
These independent vectors can be linearly combined to form all possible
vectors in the vector space and are referred to as the _basis_ of the
vector space - they are said to _span_ the space.

In signal processing, we want to change our signal into a linear
combination of a basis of independent vectors for us to then remove the
independent vectors that contribute the least to the signal. The trick
is to find the most suitable basis that allows us to compress the signal
or remove noise without too badly degrading the quality of the signal.

The best bases are orthogonal as their inverses are simply the transpose
(or complex conjugate transpose) of the matrix containing them. The two
that I have so far considered are the Fourier basis and the Haar wavelet
basis.

### Fourier

The Fourier transform involves changing our signal into a linear
combination of independent vectors that each represent a different
frequency of sinusoid that makes up the signal. The extend to which each
frequency contributes to the overall signal is provided by the coefficient
of this independent vector.

Let us create a random signal of length 64 that is formed of 3 sinusoids
of random frequency.

```python
import sp
signal = sp.signals.random_1d_signal(64, 3)
```

We can then plot this signal to see what it looks like.

```python
signal_plot = sp.plotting.plot(signal, grid=True)
```

![alt text][fourier_signal_plot]

Now perform the inverse Discete Fourier Transform on the signal to obtain
a vector of coefficients representing the relative contributions of each
independent vectors in the Fourier basis. (There will be 64 independent
vectors in this basis.)

Plot the coefficients.

```python
coeffs = sp.bases.fourier.idft(signal)
frequency_plot = sp.plotting.plot(coeffs, grid=True)
``` 

![alt text][fourier_frequency_plot]

We can see 6 peaks in the plot above and a symmetry about 32 on the 
dimensions axis. This is because the Fourier basis is a complex basis.
We ignore the symmetry and ackowledge that there are 3 peaks for the 3
frequencies making up the original signal. All the other values are
zero.

Therefore, if we were to transmit this signal it would be faster
to transmit a smaller vector of the locations and values of the peaks 
in the plot and reconstruct it using the Fourier basis once it has been
received rather than send the much longer original signal.

We could also remove the frequencies that we believe are unwanted and 
reconstruct the signal without these components. Let's do that below for
the frequencies that have values between 5 and 10, using numpy.

```python
import numpy as np
coeffs = np.where((coeffs > 5) & (coeffs < 10), 0, coeffs)
frequency_plot.send(coeffs)
```

![alt text][fourier_frequency_plot_attenuated]

And apply the Discret Fourier Transform to the modified coefficients to
reconstruct the signal without these frequencies and plot (in orange).

```python
modified = sp.bases.fourier.dft(coeffs)
signal_plot.send(modified)
```

![alt text][fourier_signal_plot_modified]


### Wavelets

#### Haar 


[fourier_signal_plot]: https://github.com/willgraham0/signal_processing/tree/image_upload/images/fourier_signal_plot.png "fourier_signal_plot"
[fourier_frequency_plot]: https://github.com/willgraham0/signal_processing/tree/image_upload/images/fourier_frequency_plot.png "fourier_frequency_plot"
[fourier_frequency_plot_attenuated]: https://github.com/willgraham0/signal_processing/tree/image_upload/images/fourier_frequency_plot.png "fourier_frequency_plot_attenuated"
[fourier_signal_plot_modified]: https://github.com/willgraham0/signal_processing/tree/image_upload/images/fourier_signal_plot.png "fourier_signal_plot_modified"
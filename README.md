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
combination of a basis of independent vectors for us then to remove the
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
frequency of sinusoid that makes up the signal. The extent to which each
frequency contributes to the overall signal is provided by the coefficient
of each independent vector.

#### A Simple Fourier Transformation

Let us create a signal of length 64 that is formed of 3 sinusoids
of random frequency and amplitude.

```python
import sp
signal = sp.signals.sum_of_sinusoids(64, 3)
```

We can then plot this signal to see what it looks like.

```python
signal_plot = sp.plotting.plot(signal, grid=True)
```

![alt text][fourier_signal_plot]

Now perform the inverse Discete Fourier Transform on the signal to obtain
a vector of coefficients representing the relative contributions of each
independent vector in the (inverse) Fourier basis. (There will be 64
independent vectors in this basis.)

```python
coeffs = sp.bases.fourier.idft(signal)
frequency_plot = sp.plotting.plot(coeffs, grid=True)
```

And plot the vector of coefficients.

![alt text][fourier_frequency_plot]

We can see 6 peaks in the plot above and a symmetry about 32 on the 
dimensions axis. This is because the (inverse) Fourier basis is a complex
basis. We ignore the symmetry and ackowledge that there are 3 peaks for
the 3 frequencies making up the original signal. All the other values are
zero.

Therefore, if we were to transmit this signal it would be faster than
sending the much longer original signal - we are only transmitting small
vector of the locations and values of the peaks in the plot and reconstructing
it using the Fourier basis once it has been received.

We could also remove the frequencies that we believe are unwanted and 
reconstruct the signal without these components. Let's do that below for
the frequencies that have values between 5 and 10, using numpy.

```python
import numpy as np
coeffs = np.where((coeffs > 5) & (coeffs < 10), 0, coeffs)
frequency_plot.send(coeffs)
```

![alt text][fourier_frequency_plot_attenuated]

Apply the Discret Fourier Transform to the modified coefficients to
reconstruct the signal without these frequencies and plot (in orange).

```python
modified = sp.bases.fourier.dft(coeffs)
signal_plot.send(modified)
```

![alt text][fourier_signal_plot_modified]

#### Image Frequency Modulation

The Fourier transformation can be extended to signals of higher
dimensions.


#### The Problem with the Fourier basis

As discussed, the Fourier bases is one that is made up of independent
vectors that represent different frequencies of sinusoids that all sum
to make a signal. In the first example the signal was a sum of 3
frequencies of sinusoid, which means that only 3 of the 64 independent
vectors were needed. Some signals, however, are not made up of only 2 or 3
frequencies but actually all the indepentent vectors that made up the
space of, in that example, 64 dimensions.

Let's make a square wave, this time of length 100, and plot it.

```python
square = sp.signals.square_signal(100)
signal_plot = sp.plotting.plot(square, grid=True)
``` 

![alt text][square_signal_plot]

Now let's find it's coefficients and plot those.

```python
coeffs = sp.bases.fourier.idft(square)
frequency_plot = sp.plotting.plot(coeffs)
```

![alt text][square_frequency_plot]

We can see that we now have 100 non-zero values - 100 non-zero coefficients
of independent vectors in the (inverse) Fourier basis. It is going to be
difficult to compress this signal, but let's try. Set coefficients with
values less than 0.1 and greater than -0.1 to zero. The new coefficients
are in orange.

```python
import numpy as np
coeffs = np.where((coeffs > -0.1) & (coeffs < 0.1), 0, coeffs)
frequency_plot.send(coeffs)
```

![alt text][square_frequency_plot_attenuated]

And reassembling the signal from these modified coefficients produces...

```python
modified = sp.bases.fourier.dft(coeffs)
signal_plot.send(modified)
```

![alt text][square_signal_plot_modified]

As you can see, compression by removing the independent vectors with
coefficients between -0.1 and 0.1 has produced a poor result on the
square wave. We can do better for signals such as these - using wavelets!

### Wavelets

#### Haar 


[fourier_signal_plot]: images/fourier_signal_plot.png "fourier_signal_plot"
[fourier_frequency_plot]: images/fourier_frequency_plot.png "fourier_frequency_plot"
[fourier_frequency_plot_attenuated]: images/fourier_frequency_plot_attenuated.png "fourier_frequency_plot_attenuated"
[fourier_signal_plot_modified]: images/fourier_signal_plot_modified.png "fourier_signal_plot_modified"
[square_signal_plot]: images/square_signal_plot.png "square_signal_plot"
[square_frequency_plot]: images/square_frequency_plot.png "square_frequency_plot"
[square_frequency_plot_attenuated]: images/square_frequency_plot_attenuated.png "square_frequency_plot_attenuated"
[square_signal_plot_modified]: images/square_signal_plot_modified.png "square_signal_plot_modified"
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
(or complex conjugate transpose) of the matrix containing them. This means
that the transformation of the signal into its combination of independent
vectors can be reversed by the use of this transposed basis matrix.

The two that I have so far considered are the Fourier basis and the Haar
wavelet basis.

### Fourier

The Fourier transform involves changing our signal into a linear
combination of independent vectors that each represents a different
frequency of sinusoid that makes up the signal. The extent to which each
frequency contributes to the overall signal is provided by the coefficient
of each independent vector.

#### A Simple Fourier Transformation

Let us create a signal of length 64 that is formed of 3 sinusoids
of amplitudes, 3, 8 and 2, and frequencies, 17, 26 and 29.

```python
import sp
signal = sp.signals.sum_of_sinusoids(64, [[3, 17], [8, 26], [2, 29]])
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

Therefore, if we were to transmit this signal of three sinusoids, it would
be faster to send a small vector of the locations and values of the peaks
in the frequency plot (which we have in the _coeffs_ variable) and
reconstruct it using the Fourier basis once it has been received 
than it is to send the much longer original signal. We have performed
lossless compression of the signal.

Additionally, we could remove frequencies that we believe are unwanted and 
reconstruct the signal without these components (lossy compression).
Let's do that below for the frequencies that have values between 5 and
10, using numpy.

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

#### Fourier Transformation of Images

The Fourier transformation can be extended to signals of higher
dimensions. If we consider images as our signal and perform the (inverse)
Discrete Fourier Transform on this we can find the frequencies that make
up in the image along its two axes.

Let's make a 100x100 image consisting of two 2-dimenisonal sinusoids of two
frequencies in the horizontal direction and two 2-dimensional sinusoids
of two frequencies in the vertical direction and then visualise it.

Note:
Horizontal sinusoids means that the sinusoid rotates horizontally, i.e.
vertical stripes will appear in the 2-d image.

```python
# Horizontal sinosoids of low (2 rads/sec) and high (5 rads/sec) freqency)
horiz_low = sp.signals.horizontal_sinusoids(100, 100, 2)
horiz_high = sp.signals.horizontal_sinusoids(100, 100, 5)

# Vertical sinosoids of low (2 rads/sec) and high (5 rads/sec) freqency)
vert_low = sp.signals.vertical_sinusoids(100, 100, 2)
vert_high = sp.signals.vertical_sinusoids(100, 100, 5)

# Add all together to make the signal
signal = horiz_low + horiz_high + vert_low + vert_high
sp.plotting.plot(signal)
```

![alt text][fourier_signal_plot_2d]

Now perform the inverse 2-d Discete Fourier Transform on the image and
plot the 2d array of Fourier coefficients.

```python
coeffs = sp.bases.fourier.idft2(signal)
sp.plotting.plot(coeffs)
```

![alt text][fourier_frequency_plot_2d]

In the top left of the frequency plot above, we can see four black points.
These represent the four frequencies that make up the image (two in the
horizontal direction at 2 and 5 and two in the vertical direction at 2
and 5). As before, we have symmetry about the centre of the image along
both axes due to the complex nature of the Fourier bases.

If we were to transmit this image it would be faster to send the
coefficients and their locations and reconstruct the image at the other
end that it would be to send all the information. This is lossless
image compression.

Again, we can remove frequencies that we believe are noise but this time,
instead of removing those that contribute least to the image (i.e. have
small coefficients), we will remove the higher frequency components of the
image (in both directions) that we know to exist. This is lossy
compression as we are reducing the complexity of the image but which
cannot be restored by reconstruction of the image from the remaining
coefficients. 

```python
# Vertically
coeffs[5, 0] = 0
coeffs[100-5, 0] = 0
# Horizontally
coeffs[0, 5] = 0
coeffs[0, 100-5] = 0
sp.plotting.plot(coeffs)
```

![alt text][fourier_frequency_plot_2d_attenuated]

And apply the 2-dimensional Discret Fourier Transform to the modified
coefficients to reconstruct the signal without these high frequencies
and plot.

```python
modified = sp.bases.fourier.dft(coeffs)
sp.plotting.plot(modified)
```

![alt text][fourier_signal_plot_2d_modified]

#### The Problem with the Fourier Basis

As discussed, the Fourier bases is one that is made up of independent
vectors that represent different frequencies of sinusoids that sum
to make a signal. In the first example the signal was a sum of 3
frequencies of sinusoid, which means that only 3 of the 64 independent
vectors were needed. Some signals, however, are not made up of such a
small number of frequencies but actually all the possible frequencies -
all the possible indepentent vectors that made up the vector space (64
in that example).

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
square wave. We can do better for signals such as these by using better
bases. These alternative bases are called wavelets!

### Wavelets

Unlike the Fourier basis that is made up of independent vectors that
each represent a different frequency over the entire length of a signal,
the wavelet basis is made up of independent vectors that each represent
a frequency (or number of frequencies) over a certain period of the
signal. An analogy can be made with written music; the horizontal length
of the staff is the time over which a note may be played and the
vertical position of the notes on the lines and spaces of the staff is
the frequency at which the notes will be played.

The wavelet transform involves changing our signal into a linear
combination of independent wavelet vectors. The extent to which each
vector contributes to the overall signal is provided by the coefficient
of each independent vector.

There are a number of different wavelet bases and the selection of one
of them, including the selection of wavelets over the Fourier basis,
depends on what you are trying to achieve (for example, signal
compression) and the characteristics of your signal. We will discuss
this here. 

#### Haar 

Haar wavelets are the simplest of the wavelets. They consist of a
single, complete square wave that is dilated (stretched) and translated
(shifted) across the entire length of the signal.

The orthonormal 4x4 Haar wavelet matrix is:

```python
haar_matrix = sp.bases.wavelets.Haar.matrix(4)

>>> haar_matrix
array([[ 0.5  ,  0.5  ,  0.707,  0.   ],
       [ 0.5  ,  0.5  , -0.707,  0.   ],
       [ 0.5  , -0.5  ,  0.   ,  0.707],
       [ 0.5  , -0.5  ,  0.   , -0.707]])
```

Each column, except for column 0, is the single, complete square wave
that has been dilated, translated or both. Column 1 is the original
square wave. Column 2 is a dilated (squashed) square wave located over
the first half of the signal. Column 3 is a square wave that has the
same amount of dilation as column 2 but has been translated to the
second half of the signal.

The more the square wave is dilated (squashed) the more it represents a
signal of a higher frequency. The translation represents where that
signal of higher frequency acts. Again, like a piece of music, the
dilation represents the vertical position of the notes on the staff and
the translation represents the where the notes should be played in time -
the horizontal position. 

Let's see some examples. First, we will create a flat, dc, signal that
is 8 dimensions long and has values of 1. Then, we will perform the
wavelet transform and look at the coefficients.

```python
signal = np.ones(8)
coeffs = sp.bases.wavelets.idwt(signal, 'Haar')
>>> coeffs
array([2.828, 0.  , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ])
```

What the coefficients are telling us is that it is only column 0 of the
8x8 Haar matrix that is contributing to the signal, which is exactly
what we would expect. Now, lets add a square wave onto the dc signal
and perform the transform again.

```python
signal += np.array([  1,  1,  1,  1, -1, -1, -1, -1])
coeffs = sp.bases.wavelets.idwt(signal, 'Haar')
>>> coeffs
array([2.828, 2.828, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ])
```

Now, the coefficients are telling us that our signal is made up of a dc
signal and an undilated square wave across the whole length of the
signal, which is exactly what we added to our first signal. Now, we can
add the most dilated and most transformed Haar wavelet onto our signal
and expect the last coefficient in the array to become non-zero.

```python
signal += np.array([  0,  0,  0,  0,  0,  0,  1, -1])
coeffs = sp.bases.wavelets.idwt(signal, 'Haar')
>>> coeffs
array([2.828, 2.828, 0.   , 0.   , 0.   , 0.   , 0.   , 1.414])
```

Now let's take a random example, visualise the Haar transformation in a
different way and remove 

##### The Benefits of the Haar Wavelet Basis.

```python
square = sp.signals.square_signal(100)
signal_plot = sp.plotting.plot(square, grid=True)
```

[fourier_signal_plot]: images/fourier_signal_plot.png "fourier_signal_plot"
[fourier_frequency_plot]: images/fourier_frequency_plot.png "fourier_frequency_plot"
[fourier_frequency_plot_attenuated]: images/fourier_frequency_plot_attenuated.png "fourier_frequency_plot_attenuated"
[fourier_signal_plot_modified]: images/fourier_signal_plot_modified.png "fourier_signal_plot_modified"
[fourier_signal_plot_2d]: images/fourier_signal_plot_2d.png "fourier_signal_plot_2d"
[fourier_frequency_plot_2d]: images/fourier_frequency_plot_2d.png "fourier_frequency_plot_2d"
[fourier_frequency_plot_2d_attenuated]: images/fourier_frequency_plot_2d_attenuated.png "fourier_frequency_plot_2d_attenuated"
[fourier_signal_plot_2d_modified]: images/fourier_signal_plot_2d_modified.png "fourier_signal_plot_2d_modified"
[square_signal_plot]: images/square_signal_plot.png "square_signal_plot"
[square_frequency_plot]: images/square_frequency_plot.png "square_frequency_plot"
[square_frequency_plot_attenuated]: images/square_frequency_plot_attenuated.png "square_frequency_plot_attenuated"
[square_signal_plot_modified]: images/square_signal_plot_modified.png "square_signal_plot_modified"

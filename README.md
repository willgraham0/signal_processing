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

Let us create a random signal of length 16 that is formed of 3 sinusoids
of random frequency and amplitude.

```
import sp
signal = sp.signals.random_1d_signal(16, 3)
```

### Wavelets

#### Haar 
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

The best bases are orthogonal and the two that I have so far considered
are the Fourier basis and the Haar wavelet basis.

### Fourier

The Fourier transform involves changing our signal into a linear
combination of independent vectors that each represent a different
frequency of sinusoid. The extend to which each frequency contributes to
the overall signal is provided by its amplitude, which is the coefficient
corresponding to this independent vector.

### Wavelets

#### Haar 
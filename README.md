# Signal Processing

## Introduction

The purpose of this project is to better understand Fourier and Wavelet
transformations, which are commonly used in applications such as signal
compression and noise removal.

The code has not been written with speed in mind - there are many good
libraries out there that do this stuff fast for real applications - but
I hope for the purposes of illustrating the theory, that it does the job.

## The Basis of a Vector Space

Any 1-dimensional signal or 2-dimensional image can be thought of as a
high dimensional _vector_ existing in a high dimensional _vector space_.
It can be also be thought of as a linear combination of other high
dimensional vectors that lie in the same space and which are _indepedent_
of one another - vectors that cannot be formed by the linear combination
of the others. These independent vectors can be linearly combined to
form all possible vectors in the vector space and are referred to as the
_basis_ of the vector space - they _span_ the space.

For signal compression, the goal is to find the right basis that allows
us to remove the contributions of the independent vectors that contribute
the least to the signal. These bases are orthogonal.

### Fourier Transformations


### Wavelet Transformations

#### Haar 
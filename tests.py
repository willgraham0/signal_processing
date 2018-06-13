"""This module contains unit tests to verify the behaviour of the
signal processing functions.
"""

import unittest
import bases
import numpy as np
import signals


class FourierTests(unittest.TestCase):
    """Test cases for the Fourier basis.
    """

    def test_inverse_is_transpose(self):
        """Test if the inverse of the Fourier matrix (real) is the same
        as the transpose of the Fourier matrix (real). This should be
        true for an orthogonal basis, which the Fourier basis is.
        """
        fourier_matrix = bases.fourier.fourier_matrix(10)
        ifourier_matrix = bases.fourier.ifourier_matrix(10)
        np.testing.assert_almost_equal(fourier_matrix.T.real, ifourier_matrix.real)


    def test_inverse_is_inverse(self):
        """Test if the inverse of the Fourier matrix (real) is the same
        as the numpy.linalg.inv of the Fourier matrix (real).
        """
        fourier_matrix = bases.fourier.fourier_matrix(10)
        np_ifourier_matrix = np.linalg.inv(fourier_matrix)
        ifourier_matrix = bases.fourier.ifourier_matrix(10)
        np.testing.assert_almost_equal(ifourier_matrix.real, np_ifourier_matrix.real)


    def test_1d_signal_analysis_and_synthesis(self):
        """Test if a 1-dimensional signal (real) is synthesised back to
        its original (real) after analysis (into Fourier domain).
        """
        original = signals.random_1d_signal(20, 5)
        coeffs = bases.fourier.dft(original)
        synthesised = bases.fourier.idft(coeffs)
        np.testing.assert_almost_equal(original.real, synthesised.real)


class HaarTests(unittest.TestCase):
    """Test cases for the Haar basis.
    """

    def test_haar_wavelet_length_error(self):
        """Test if ValueError is raised if the wavelet length is greater
        than the signal length.
        """
        with self.assertRaises(ValueError):
            bases.wavelets.haar.haar_wavelet(8, 6)


    def test_haar_wavelet_log2_values(self):
        """Test if ValueError is raised if the base 2 logarithms of the
        wavelet length or signal length is not an integer.
        """
        with self.assertRaises(ValueError):
            bases.wavelets.haar.haar_wavelet(2, 10)
        with self.assertRaises(ValueError):
            bases.wavelets.haar.haar_wavelet(3, 8)

        
    def test_haar_wavelets(self):
        """Test the Haar wavelets for a range of wavelet lengths and
        signal lengths.
        """
        np.testing.assert_equal(
            bases.wavelets.haar.haar_wavelet(2, 2),
            np.array([ 1, -1])
        )
        np.testing.assert_equal(
            bases.wavelets.haar.haar_wavelet(2, 4),
            np.array([ 1, -1,  0,  0])
        )
        np.testing.assert_equal(
            bases.wavelets.haar.haar_wavelet(4, 4),
            np.array([ 1,  1, -1, -1])
        )
        np.testing.assert_equal(
            bases.wavelets.haar.haar_wavelet(2, 8),
            np.array([ 1, -1,  0,  0,  0,  0,  0,  0])
        )
        np.testing.assert_equal(
            bases.wavelets.haar.haar_wavelet(4, 8),
            np.array([ 1,  1, -1, -1,  0,  0,  0,  0])
        )
        np.testing.assert_equal(
            bases.wavelets.haar.haar_wavelet(8, 8),
            np.array([ 1,  1,  1,  1, -1, -1, -1, -1])
        )


if __name__ == "__main__":
    unittest.main()
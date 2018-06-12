"""This module contains unit tests to verify the behaviour of the
signal processing functions."""


import unittest
import bases
import numpy as np
import signals


class FourierTests(unittest.TestCase):

    def test_inverse_is_transpose(self):
        """Test if the inverse of the Fourier matrix (real) is the same
        as the transpose of the Fourier matrix (real). This should be
        true for an orthogonal basis, which the Fourier basis is."""
        dft_matrix = bases.fourier.dft_matrix(10)
        idft_matrix = bases.fourier.idft_matrix(10)
        np.testing.assert_almost_equal(dft_matrix.T.real, idft_matrix.real)


    def test_1d_signal_analysis_and_synthesis(self):
        """Test if a 1-dimensional signal (real) is synthesised back to
        its original (real) after analysis (into Fourier domain)."""
        original = signals.random_1d_signal(20, 5)
        coeffs = bases.fourier.dft(original)
        synthesised = bases.fourier.idft(coeffs)
        np.testing.assert_almost_equal(original.real, synthesised.real)


if __name__ == "__main__":
    unittest.main()
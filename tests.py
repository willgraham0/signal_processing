"""This module contains unit tests to verify the behaviour of the
signal processing functions.
"""

import unittest
from sp import bases, signals
import numpy as np


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

    def test_wavelet_length_error(self):
        """Test if ValueError is raised if the wavelet length is greater
        than the signal length.
        """
        with self.assertRaises(ValueError):
            bases.wavelets.Haar.wavelet(8, 6)


    def test_wavelet_log2_values(self):
        """Test if ValueError is raised if the base 2 logarithms of the
        wavelet length or signal length is not an integer.
        """
        with self.assertRaises(ValueError):
            bases.wavelets.Haar.wavelet(2, 10)
        with self.assertRaises(ValueError):
            bases.wavelets.Haar.wavelet(3, 8)

        
    def test_wavelets(self):
        """Test the Haar wavelets for a range of wavelet lengths and
        signal lengths.
        """
        np.testing.assert_equal(
            bases.wavelets.Haar.wavelet(2, 2),
            (1/np.sqrt(2))*np.array([ 1, -1])
        )
        np.testing.assert_equal(
            bases.wavelets.Haar.wavelet(1, 4),
            (1/np.sqrt(4))*np.array([ 1,  1,  1,  1])
        )
        np.testing.assert_equal(
            bases.wavelets.Haar.wavelet(2, 4),
            (1/np.sqrt(2))*np.array([ 1, -1,  0,  0])
        )
        np.testing.assert_equal(
            bases.wavelets.Haar.wavelet(4, 4),
            (1/np.sqrt(4))*np.array([ 1,  1, -1, -1])
        )
        np.testing.assert_equal(
            bases.wavelets.Haar.wavelet(2, 8),
            (1/np.sqrt(2))*np.array([ 1, -1,  0,  0,  0,  0,  0,  0])
        )
        np.testing.assert_equal(
            bases.wavelets.Haar.wavelet(4, 8),
            (1/np.sqrt(4))*np.array([ 1,  1, -1, -1,  0,  0,  0,  0])
        )
        np.testing.assert_equal(
            bases.wavelets.Haar.wavelet(8, 8),
            (1/np.sqrt(8))*np.array([ 1,  1,  1,  1, -1, -1, -1, -1])
        )

    
    def test_matrix(self):
        """Test that the Haar matrix is correctly produced for different
        dimensions."""
        np.testing.assert_equal(
            bases.wavelets.Haar.matrix(2),
            (1/np.sqrt(2))*np.array([[ 1,  1],
                                     [ 1, -1]])
        )
        np.testing.assert_equal(
            bases.wavelets.Haar.matrix(4),
            np.array([[ 1/np.sqrt(4),  1/np.sqrt(4),  1/np.sqrt(2),  0],
                      [ 1/np.sqrt(4),  1/np.sqrt(4), -1/np.sqrt(2),  0],
                      [ 1/np.sqrt(4), -1/np.sqrt(4),  0,  1/np.sqrt(2)],
                      [ 1/np.sqrt(4), -1/np.sqrt(4),  0, -1/np.sqrt(2)]])
        )


    def test_inverse_is_inverse(self):
        """Test if the inverse of the Haar matrix is the same as the
        numpy.linalg.inv of the Haar matrix.
        """
        haar_matrix = bases.wavelets.Haar.matrix(16)
        ihaar_matrix = bases.wavelets.Haar.imatrix(16)
        np_ihaar_matrix = np.linalg.inv(haar_matrix)
        np.testing.assert_almost_equal(ihaar_matrix, np_ihaar_matrix)


    def test_1d_signal_invalid_length(self):
        """Test if a ValueError exception is raised is the signal length
        log2(m) is not an integer for the Haar discrete wavelet
        transform. 
        """
        original = signals.random_1d_signal(17, 5)
        with self.assertRaises(ValueError):
            coeffs = bases.wavelets.dwt(original, 'Haar')


    def test_1d_signal_analysis_and_synthesis(self):
        """Test if a 1-dimensional signal (real) is synthesised back to
        its original (real) after analysis (into Haar domain).
        """
        original = signals.random_1d_signal(16, 5)
        coeffs = bases.wavelets.dwt(original, 'Haar')
        synthesised = bases.wavelets.idwt(coeffs, 'Haar')
        np.testing.assert_almost_equal(original.real, synthesised.real)


class GeneralTests(unittest.TestCase):
    """Test cases for general functionality.
    """

    def test_is_implemented(self):
        """Test that the check for provided wavelet families.
        """
        self.assertTrue(bases.wavelets.is_implemented('Haar'))
        self.assertFalse(bases.wavelets.is_implemented('Crazy'))

    
    def test_get_family(self):
        """Test that the wavelet family object is returned else a
        NotImplementedError.
        """
        self.assertEqual(bases.wavelets.get_family('Haar'),
                         bases.wavelets.Haar)
        with self.assertRaises(NotImplementedError):
            bases.wavelets.get_family('Crazy')


if __name__ == "__main__":
    unittest.main()
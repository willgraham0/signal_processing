"""This module contains unit tests to verify the behaviour of the
signal processing functions.
"""

import unittest

import numpy as np

from sp import bases, signals


class FourierTests(unittest.TestCase):
    """Test cases for the Fourier basis."""

    def test_inverse_is_transpose(self):
        """Test if the inverse of the Fourier matrix is the same as the
        transpose of the complex conjugate of the Fourier matrix. This
        should be true for an orthogonal, complex basis, which is true
        of the Fourier basis.
        """
        fourier_matrix = bases.fourier.fourier_matrix(10)
        ifourier_matrix = bases.fourier.ifourier_matrix(10)
        np.testing.assert_almost_equal(fourier_matrix.conj().T, ifourier_matrix)

    def test_inverse_is_inverse(self):
        """Test if the inverse of the Fourier matrix is the same as the
        numpy.linalg.inv of the Fourier matrix.
        """
        fourier_matrix = bases.fourier.fourier_matrix(10)
        np_ifourier_matrix = np.linalg.inv(fourier_matrix)
        ifourier_matrix = bases.fourier.ifourier_matrix(10)
        np.testing.assert_almost_equal(ifourier_matrix, np_ifourier_matrix)

    def test_1d_signal_analysis_and_synthesis(self):
        """Test if a 1-dimensional signal (real) is synthesised back to
        its original (real) after analysis (into Fourier domain).
        """
        original = signals.sum_of_sinusoids(20, [[10, 5], [5, 8]])
        coeffs = bases.fourier.idft(original)
        synthesised = bases.fourier.dft(coeffs)
        np.testing.assert_almost_equal(original.real, synthesised.real)

    def test_diag_inverse_is_transpose(self):
        """Test if the inverse of the Fourier matrix (Diagonal only) is
        the same as the transpose of the complex conjugate of the
        Fourier matrix (Diagonal only).
        """
        diag_fmatrix = bases.fourier.diagonal(10)
        idiag_fmatrix = bases.fourier.idiagonal(10)
        np.testing.assert_almost_equal(diag_fmatrix.conj().T, idiag_fmatrix)        

    def test_dimensionality_correct(self):
        """Test if ValueError is raised if a 1-dimensional signal is 
        passed to dft2 or idft2 and if a 2-dimensional signal is passed 
        to dft or idft. 
        """
        one_dimen = np.random.rand(50)
        two_dimen = np.random.rand(50, 50)
        with self.assertRaises(ValueError):
            bases.fourier.idft(two_dimen)
            bases.fourier.dft(two_dimen)
            bases.fourier.idft2(one_dimen)
            bases.fourier.dft2(one_dimen)


class HaarTests(unittest.TestCase):
    """Test cases for the Haar basis."""

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
            np.multiply(np.array([[ 1,  1,  1,  0],
                                  [ 1,  1, -1,  0],
                                  [ 1, -1,  0,  1],
                                  [ 1, -1,  0, -1]]),
                        np.array([1/np.sqrt(4),
                                  1/np.sqrt(4),
                                  1/np.sqrt(2),
                                  1/np.sqrt(2)]))
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
        original = signals.sum_of_sinusoids(17, [[10, 2],[5, 5]])
        with self.assertRaises(ValueError):
            coeffs = bases.wavelets.idwt(original, 'Haar')

    def test_1d_signal_analysis_and_synthesis(self):
        """Test if a 1-dimensional signal (real) is synthesised back to
        its original (real) after analysis (into Haar domain).
        """
        original = signals.sum_of_sinusoids(16, [[10, 2],[5, 5]])
        coeffs = bases.wavelets.idwt(original, 'Haar')
        synthesised = bases.wavelets.dwt(coeffs, 'Haar')
        np.testing.assert_almost_equal(original.real, synthesised.real)

    def test_compress(self):
        """Test the compression of a Haar wavelet matrix into one with
        no zeros.
        """
        haar_matrix = bases.wavelets.Haar.matrix(8)
        squeezed = bases.wavelets.Haar.squeeze(haar_matrix)
        true_squeezed = np.multiply(np.array([[1,  1,  1,  1],
                                              [1,  1,  1, -1],
                                              [1,  1, -1,  1],
                                              [1,  1, -1, -1],
                                              [1, -1,  1,  1],
                                              [1, -1,  1, -1],
                                              [1, -1, -1,  1],
                                              [1, -1, -1, -1]]),
                                    np.array([1/np.sqrt(8),
                                              1/np.sqrt(8),
                                              1/np.sqrt(4),
                                              1/np.sqrt(2)]))
        np.testing.assert_almost_equal(squeezed, true_squeezed)


class GeneralTests(unittest.TestCase):
    """Test cases for general functionality."""

    def test_is_implemented(self):
        """Test that the check for provided wavelet families."""
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


class SignalsTests(unittest.TestCase):
    """Test cases for signals functionality."""

    def test_signals_dimensions(self):
        """Test that the signals have the correct dimensionality."""
        self.assertEqual(signals.sum_of_sinusoids(30, [[10, 4]]).shape, (30,))
        self.assertEqual(signals.square_signal(30).shape, (30,))
        self.assertEqual(signals.chequered(50, 30, 2).shape, (50, 30))
        self.assertEqual(signals.stripes(50, 30, 2).shape, (50, 30))
        self.assertEqual(signals.sinusoids_2d(20, 100, 2).shape, (20, 100))
        self.assertEqual(signals.vertical_sinusoids(20, 100, 2).shape, (20, 100))
        self.assertEqual(signals.horizontal_sinusoids(20, 100, 2).shape, (20, 100))


if __name__ == "__main__":
    unittest.main()
    